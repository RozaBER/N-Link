import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from typing import Tuple, Optional


class MEGMelAligner(nn.Module):
    """
    Aligns MEG signals with mel-spectrograms using contrastive learning
    Based on NEURAL-VOX approach
    """
    
    def __init__(
        self,
        meg_channels: int = 208,
        sampling_rate: int = 1000,
        mel_bins: int = 80,
        window_size_ms: int = 250,
        hop_size_ms: int = 10,
        audio_sr: int = 16000,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.meg_channels = meg_channels
        self.sampling_rate = sampling_rate
        self.mel_bins = mel_bins
        self.window_size_ms = window_size_ms
        self.hop_size_ms = hop_size_ms
        self.audio_sr = audio_sr
        self.temperature = temperature
        
        # MEG encoder
        self.meg_encoder = nn.Sequential(
            nn.Conv1d(meg_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.GELU(),
        )
        
        # Mel encoder
        self.mel_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        
        # Projection heads
        self.meg_projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
        )
        
        self.mel_projection = nn.Sequential(
            nn.Linear(128 * (mel_bins // 4), 256),
            nn.GELU(),
            nn.Linear(256, 128),
        )
        
        # Mel-spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=audio_sr,
            n_mels=mel_bins,
            n_fft=int(audio_sr * 0.025),  # 25ms window
            hop_length=int(audio_sr * 0.010),  # 10ms hop
        )
        
    def extract_mel_features(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convert audio to mel-spectrogram
        
        Args:
            audio: (B, T_audio) audio waveform
            
        Returns:
            mel_spec: (B, 1, mel_bins, T_mel) mel-spectrogram
        """
        mel_spec = self.mel_transform(audio)
        mel_spec = torch.log(mel_spec + 1e-10)
        mel_spec = mel_spec.unsqueeze(1)  # Add channel dimension
        return mel_spec
    
    def encode_meg(self, meg_signal: torch.Tensor) -> torch.Tensor:
        """
        Encode MEG signals
        
        Args:
            meg_signal: (B, C, T) MEG signals
            
        Returns:
            meg_features: (B, D) MEG feature vector
        """
        # Encode with convolutions
        meg_feat = self.meg_encoder(meg_signal)  # (B, 512, T')
        
        # Global average pooling
        meg_feat = meg_feat.mean(dim=-1)  # (B, 512)
        
        # Project to common space
        meg_feat = self.meg_projection(meg_feat)  # (B, 128)
        
        return F.normalize(meg_feat, dim=-1)
    
    def encode_mel(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Encode mel-spectrogram
        
        Args:
            mel_spec: (B, 1, mel_bins, T) mel-spectrogram
            
        Returns:
            mel_features: (B, D) mel feature vector
        """
        # Encode with 2D convolutions
        mel_feat = self.mel_encoder(mel_spec)  # (B, 128, mel_bins//4, T')
        
        # Reshape and pool
        B, C, H, W = mel_feat.shape
        mel_feat = mel_feat.permute(0, 3, 1, 2).reshape(B, W, -1)  # (B, T', C*H)
        mel_feat = mel_feat.mean(dim=1)  # (B, C*H)
        
        # Project to common space
        mel_feat = self.mel_projection(mel_feat)  # (B, 128)
        
        return F.normalize(mel_feat, dim=-1)
    
    def compute_contrastive_loss(
        self,
        meg_features: torch.Tensor,
        mel_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss
        
        Args:
            meg_features: (B, D) normalized MEG features
            mel_features: (B, D) normalized mel features
            
        Returns:
            loss: scalar contrastive loss
        """
        batch_size = meg_features.shape[0]
        
        # Compute similarity matrix
        similarity = torch.matmul(meg_features, mel_features.T) / self.temperature
        
        # Labels: positive pairs are on the diagonal
        labels = torch.arange(batch_size, device=meg_features.device)
        
        # Compute loss in both directions
        loss_meg_to_mel = F.cross_entropy(similarity, labels)
        loss_mel_to_meg = F.cross_entropy(similarity.T, labels)
        
        return (loss_meg_to_mel + loss_mel_to_meg) / 2
    
    def forward(
        self,
        meg_signal: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
        mel_spec: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass
        
        Args:
            meg_signal: (B, C, T) MEG signals
            audio: (B, T_audio) audio waveform (optional)
            mel_spec: (B, 1, mel_bins, T_mel) pre-computed mel-spectrogram (optional)
            
        Returns:
            dict with:
                - meg_features: (B, D) MEG features
                - mel_features: (B, D) mel features (if audio/mel provided)
                - loss: contrastive loss (if audio/mel provided)
        """
        # Encode MEG
        meg_features = self.encode_meg(meg_signal)
        
        output = {"meg_features": meg_features}
        
        # If audio or mel provided, compute mel features and loss
        if audio is not None or mel_spec is not None:
            if mel_spec is None:
                mel_spec = self.extract_mel_features(audio)
            
            mel_features = self.encode_mel(mel_spec)
            output["mel_features"] = mel_features
            
            # Compute contrastive loss
            loss = self.compute_contrastive_loss(meg_features, mel_features)
            output["loss"] = loss
        
        return output
    
    def align_meg_mel(
        self,
        meg_signal: torch.Tensor,
        audio: torch.Tensor,
        meg_timestamps: Optional[Tuple[float, float]] = None,
        audio_timestamps: Optional[Tuple[float, float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align MEG signals with audio using timestamps
        
        Args:
            meg_signal: (C, T) MEG signal
            audio: (T_audio,) audio waveform
            meg_timestamps: (start, end) in seconds for MEG
            audio_timestamps: (start, end) in seconds for audio
            
        Returns:
            aligned_meg: (C, T_aligned) aligned MEG
            aligned_mel: (mel_bins, T_aligned) aligned mel-spectrogram
        """
        # Extract relevant segments if timestamps provided
        if meg_timestamps:
            start_idx = int(meg_timestamps[0] * self.sampling_rate)
            end_idx = int(meg_timestamps[1] * self.sampling_rate)
            meg_signal = meg_signal[:, start_idx:end_idx]
        
        if audio_timestamps:
            start_idx = int(audio_timestamps[0] * self.audio_sr)
            end_idx = int(audio_timestamps[1] * self.audio_sr)
            audio = audio[start_idx:end_idx]
        
        # Convert audio to mel
        mel_spec = self.extract_mel_features(audio.unsqueeze(0)).squeeze(0, 1)
        
        # Align temporal dimensions by interpolation
        meg_len = meg_signal.shape[-1]
        mel_len = mel_spec.shape[-1]
        
        if meg_len != mel_len:
            # Interpolate MEG to match mel length
            meg_signal = F.interpolate(
                meg_signal.unsqueeze(0),
                size=mel_len,
                mode='linear',
                align_corners=False
            ).squeeze(0)
        
        return meg_signal, mel_spec