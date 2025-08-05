import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np


class StreamingHiFiGAN(nn.Module):
    """
    Streaming-capable HiFi-GAN vocoder for real-time audio synthesis
    Adapted for MEG-to-audio generation with chunk-based processing
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        upsample_rates: List[int] = [8, 8, 2, 2],
        upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        num_mels: int = 80,
        chunk_size_ms: int = 50,
        overlap_ms: int = 10,
        sampling_rate: int = 16000,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_mels = num_mels
        self.chunk_size_ms = chunk_size_ms
        self.overlap_ms = overlap_ms
        self.sampling_rate = sampling_rate
        
        # Input projection
        self.input_conv = nn.Conv1d(input_dim, 512, 7, padding=3)
        
        # Upsampling layers
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        
        channels = 512
        for i, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(
                    channels,
                    channels // 2,
                    kernel,
                    stride=rate,
                    padding=(kernel - rate) // 2,
                )
            )
            
            # Residual blocks for each upsampling layer
            resblock_list = nn.ModuleList()
            for kernel_size, dilation_sizes in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                resblock_list.append(
                    ResBlock(channels // 2, kernel_size, dilation_sizes)
                )
            self.resblocks.append(resblock_list)
            
            channels = channels // 2
        
        # Output layers
        self.post_conv = nn.Conv1d(channels, 1, 7, padding=3, bias=False)
        self.tanh = nn.Tanh()
        
        # Streaming buffer for overlap-add
        self.register_buffer('overlap_buffer', torch.zeros(1, 1, 0))
    
    def forward(self, features: torch.Tensor, streaming: bool = False) -> torch.Tensor:
        """
        Generate audio from features
        
        Args:
            features: (B, T, D) feature sequence
            streaming: Enable streaming mode with overlap-add
            
        Returns:
            audio: (B, 1, T_audio) generated audio
        """
        # Transpose for conv layers
        x = features.transpose(1, 2)  # (B, D, T)
        
        # Initial convolution
        x = self.input_conv(x)
        
        # Progressive upsampling with residual blocks
        for i, (up, resblock_list) in enumerate(zip(self.ups, self.resblocks)):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            
            # Apply residual blocks
            res_sum = 0
            for resblock in resblock_list:
                res_sum += resblock(x)
            x = res_sum / len(resblock_list)
        
        # Final convolution
        x = F.leaky_relu(x, 0.1)
        audio = self.tanh(self.post_conv(x))
        
        if streaming:
            audio = self._apply_overlap_add(audio)
        
        return audio
    
    def _apply_overlap_add(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply overlap-add for streaming synthesis"""
        B, C, T = audio.shape
        overlap_samples = int(self.overlap_ms * self.sampling_rate / 1000)
        
        if self.overlap_buffer.size(2) > 0:
            # Add overlap from previous chunk
            audio[:, :, :overlap_samples] += self.overlap_buffer
        
        # Save overlap for next chunk
        self.overlap_buffer = audio[:, :, -overlap_samples:].clone()
        
        return audio[:, :, :-overlap_samples]
    
    def reset_streaming_buffer(self):
        """Reset the streaming buffer for new sequence"""
        self.overlap_buffer.zero_()


class ResBlock(nn.Module):
    """Residual block for HiFi-GAN"""
    
    def __init__(self, channels: int, kernel_size: int, dilation_sizes: List[int]):
        super().__init__()
        self.convs = nn.ModuleList()
        
        for dilation in dilation_sizes:
            self.convs.append(
                nn.Sequential(
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        dilation=dilation,
                        padding=(kernel_size * dilation - dilation) // 2,
                    ),
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        padding=kernel_size // 2,
                    ),
                )
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = x + conv(x)
        return x


class PhonemeDecoder(nn.Module):
    """CTC-based phoneme decoder"""
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        num_phonemes: int = 70,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Projection layer
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # LSTM layers for temporal modeling
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_phonemes),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode phonemes from features
        
        Args:
            features: (B, T, D) feature sequence
            
        Returns:
            phoneme_logits: (B, T, num_phonemes) phoneme predictions
        """
        # Project input
        x = self.input_proj(features)
        
        # LSTM encoding
        x, _ = self.lstm(x)
        
        # Output projection
        phoneme_logits = self.output_proj(x)
        
        return phoneme_logits


class MultiOutputDecoder(nn.Module):
    """
    Multi-task decoder for text, phonemes, and audio generation
    Coordinates outputs from LLaVA with specialized decoders
    """
    
    def __init__(
        self,
        llava_hidden_dim: int = 4096,
        shared_dim: int = 768,
        num_phonemes: int = 70,
        vocab_size: int = 32000,  # LLaVA default vocab
        audio_codec_dim: int = 80,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.llava_hidden_dim = llava_hidden_dim
        self.shared_dim = shared_dim
        
        # Shared projection from LLaVA hidden states
        self.shared_projection = nn.Sequential(
            nn.Linear(llava_hidden_dim, shared_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim * 2, shared_dim),
            nn.LayerNorm(shared_dim),
        )
        
        # Text decoder (can use LLaVA's language head)
        self.text_head = nn.Linear(shared_dim, vocab_size)
        
        # Phoneme decoder with CTC
        self.phoneme_decoder = PhonemeDecoder(
            input_dim=shared_dim,
            hidden_dim=512,
            num_phonemes=num_phonemes,
            dropout=dropout,
        )
        
        # Audio synthesizer
        self.audio_synthesizer = StreamingHiFiGAN(
            input_dim=shared_dim,
            upsample_rates=[8, 8, 2, 2],
            chunk_size_ms=50,
        )
        
        # Cross-modal alignment layers
        self.text_audio_align = nn.MultiheadAttention(
            embed_dim=shared_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
        
        self.phoneme_audio_align = nn.MultiheadAttention(
            embed_dim=shared_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
    
    def forward(
        self,
        llava_hidden_states: torch.Tensor,
        llava_outputs: Optional[Dict] = None,
        target_modalities: List[str] = ["text", "phoneme", "audio"],
        streaming: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate multiple output modalities
        
        Args:
            llava_hidden_states: (B, T, D) hidden states from LLaVA
            llava_outputs: Optional LLaVA model outputs
            target_modalities: List of modalities to generate
            streaming: Enable streaming for audio generation
            
        Returns:
            Dictionary with outputs for each modality
        """
        # Project to shared dimension
        shared_features = self.shared_projection(llava_hidden_states)
        
        outputs = {}
        
        # Text generation
        if "text" in target_modalities:
            text_logits = self.text_head(shared_features)
            outputs["text_logits"] = text_logits
        
        # Phoneme prediction
        if "phoneme" in target_modalities:
            phoneme_logits = self.phoneme_decoder(shared_features)
            outputs["phoneme_logits"] = phoneme_logits
        
        # Audio synthesis
        if "audio" in target_modalities:
            # Align features for audio generation
            audio_features, text_attention = self.text_audio_align(
                shared_features,
                shared_features,
                shared_features,
            )
            
            # Generate audio
            audio = self.audio_synthesizer(audio_features, streaming=streaming)
            outputs["audio"] = audio
            outputs["audio_attention_weights"] = text_attention
        
        # Compute alignment features if multiple modalities
        if len(target_modalities) > 1:
            outputs["alignment_features"] = self._compute_alignment_features(
                shared_features,
                outputs,
            )
        
        return outputs
    
    def _compute_alignment_features(
        self,
        shared_features: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute features for cross-modal alignment loss"""
        alignment_features = {}
        
        # Text-audio alignment
        if "text_logits" in outputs and "audio" in outputs:
            text_emb = F.softmax(outputs["text_logits"], dim=-1) @ self.text_head.weight
            audio_emb = shared_features.mean(dim=1)  # Global pooling
            alignment_features["text_audio_sim"] = F.cosine_similarity(
                text_emb.mean(dim=1),
                audio_emb,
            )
        
        # Phoneme-audio alignment
        if "phoneme_logits" in outputs and "audio" in outputs:
            phoneme_probs = F.softmax(outputs["phoneme_logits"], dim=-1)
            alignment_features["phoneme_audio_corr"] = self._compute_phoneme_audio_correlation(
                phoneme_probs,
                shared_features,
            )
        
        return alignment_features
    
    def _compute_phoneme_audio_correlation(
        self,
        phoneme_probs: torch.Tensor,
        audio_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute correlation between phoneme predictions and audio features"""
        # Simplified correlation computation
        phoneme_energy = phoneme_probs.sum(dim=-1)  # (B, T)
        audio_energy = audio_features.norm(dim=-1)  # (B, T)
        
        # Normalize
        phoneme_energy = (phoneme_energy - phoneme_energy.mean(dim=1, keepdim=True)) / (
            phoneme_energy.std(dim=1, keepdim=True) + 1e-8
        )
        audio_energy = (audio_energy - audio_energy.mean(dim=1, keepdim=True)) / (
            audio_energy.std(dim=1, keepdim=True) + 1e-8
        )
        
        # Compute correlation
        correlation = (phoneme_energy * audio_energy).mean(dim=1)
        
        return correlation
    
    def generate_outputs(
        self,
        llava_hidden_states: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        streaming: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate outputs with different decoding strategies
        
        Args:
            llava_hidden_states: Hidden states from LLaVA
            max_length: Maximum generation length
            temperature: Sampling temperature
            streaming: Enable streaming for audio
            
        Returns:
            Generated outputs for each modality
        """
        outputs = self.forward(
            llava_hidden_states,
            target_modalities=["text", "phoneme", "audio"],
            streaming=streaming,
        )
        
        # Decode text with sampling
        if "text_logits" in outputs:
            text_ids = self._sample_text(
                outputs["text_logits"],
                temperature=temperature,
                max_length=max_length,
            )
            outputs["text_ids"] = text_ids
        
        # Decode phonemes with CTC
        if "phoneme_logits" in outputs:
            phoneme_ids = self._ctc_decode(outputs["phoneme_logits"])
            outputs["phoneme_ids"] = phoneme_ids
        
        return outputs
    
    def _sample_text(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        max_length: int = 100,
    ) -> torch.Tensor:
        """Sample text tokens from logits"""
        B, T, V = logits.shape
        
        # Apply temperature
        scaled_logits = logits / temperature
        
        # Sample tokens
        probs = F.softmax(scaled_logits, dim=-1)
        tokens = torch.multinomial(probs.view(-1, V), 1).view(B, T)
        
        return tokens
    
    def _ctc_decode(self, logits: torch.Tensor) -> List[List[int]]:
        """CTC greedy decoding for phonemes"""
        # Get most likely phonemes
        predictions = logits.argmax(dim=-1)  # (B, T)
        
        decoded = []
        for pred in predictions:
            # Remove blanks and repetitions
            phonemes = []
            prev = -1
            for p in pred:
                if p != 0 and p != prev:  # 0 is blank token
                    phonemes.append(p.item())
                prev = p
            decoded.append(phonemes)
        
        return decoded