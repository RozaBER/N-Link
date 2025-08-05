import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss for MEG-Mel alignment"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss between two sets of features
        
        Args:
            features_a: (B, D) normalized features from modality A
            features_b: (B, D) normalized features from modality B
            mask: (B,) optional mask for valid samples
            
        Returns:
            loss: scalar loss value
        """
        batch_size = features_a.shape[0]
        
        # Compute similarity matrix
        similarity = torch.matmul(features_a, features_b.T) / self.temperature
        
        # Apply mask if provided
        if mask is not None:
            mask_matrix = mask.unsqueeze(1) * mask.unsqueeze(0)
            similarity = similarity.masked_fill(~mask_matrix, -float('inf'))
        
        # Labels: positive pairs are on the diagonal
        labels = torch.arange(batch_size, device=features_a.device)
        
        # Compute loss in both directions
        loss_a_to_b = F.cross_entropy(similarity, labels)
        loss_b_to_a = F.cross_entropy(similarity.T, labels)
        
        return (loss_a_to_b + loss_b_to_a) / 2


class CLIPLoss(nn.Module):
    """CLIP-style contrastive loss for brain-text alignment"""
    
    def __init__(self, temperature: float = 0.1, label_smoothing: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        brain_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute CLIP loss with label smoothing
        
        Args:
            brain_features: (B, D) brain encoder features
            text_features: (B, D) text encoder features
            
        Returns:
            Dictionary with loss components
        """
        # Normalize features
        brain_features = F.normalize(brain_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute logits
        logits_per_brain = brain_features @ text_features.T / self.temperature
        logits_per_text = logits_per_brain.T
        
        batch_size = brain_features.shape[0]
        labels = torch.arange(batch_size, device=brain_features.device)
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            loss_brain = self._cross_entropy_with_smoothing(logits_per_brain, labels)
            loss_text = self._cross_entropy_with_smoothing(logits_per_text, labels)
        else:
            loss_brain = F.cross_entropy(logits_per_brain, labels)
            loss_text = F.cross_entropy(logits_per_text, labels)
        
        loss = (loss_brain + loss_text) / 2
        
        # Compute accuracy
        with torch.no_grad():
            pred_brain = logits_per_brain.argmax(dim=-1)
            pred_text = logits_per_text.argmax(dim=-1)
            acc_brain = (pred_brain == labels).float().mean()
            acc_text = (pred_text == labels).float().mean()
        
        return {
            "loss": loss,
            "loss_brain_to_text": loss_brain,
            "loss_text_to_brain": loss_text,
            "accuracy_brain_to_text": acc_brain,
            "accuracy_text_to_brain": acc_text,
        }
    
    def _cross_entropy_with_smoothing(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Cross entropy with label smoothing"""
        n_classes = logits.shape[-1]
        smoothing = self.label_smoothing
        
        # Create smoothed target distribution
        targets = torch.zeros_like(logits)
        targets.fill_(smoothing / (n_classes - 1))
        targets.scatter_(1, labels.unsqueeze(1), 1 - smoothing)
        
        # Compute loss
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(targets * log_probs).sum(dim=-1).mean()
        
        return loss


class CTCPhonemeLoss(nn.Module):
    """CTC loss wrapper for phoneme prediction"""
    
    def __init__(self, blank_id: int = 0, zero_infinity: bool = True):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=zero_infinity)
    
    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CTC loss for phoneme sequences
        
        Args:
            log_probs: (T, B, C) log probabilities from model
            targets: (B, S) target phoneme sequences
            input_lengths: (B,) lengths of input sequences
            target_lengths: (B,) lengths of target sequences
            
        Returns:
            loss: scalar CTC loss
        """
        # Ensure log probabilities
        if not log_probs.is_contiguous():
            log_probs = log_probs.contiguous()
        
        # Transpose to CTC format if needed
        if log_probs.dim() == 3 and log_probs.shape[0] < log_probs.shape[1]:
            log_probs = log_probs.transpose(0, 1)
        
        loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        
        return loss


class AudioReconstructionLoss(nn.Module):
    """Multi-scale audio reconstruction loss"""
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        l2_weight: float = 0.5,
        stft_weight: float = 2.0,
        mel_weight: float = 1.0,
        perceptual_weight: float = 0.1,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.stft_weight = stft_weight
        self.mel_weight = mel_weight
        self.perceptual_weight = perceptual_weight
        
        # STFT parameters for spectral loss
        self.stft_params = [
            {"n_fft": 512, "hop_length": 160, "win_length": 512},
            {"n_fft": 1024, "hop_length": 320, "win_length": 1024},
            {"n_fft": 2048, "hop_length": 640, "win_length": 2048},
        ]
    
    def forward(
        self,
        predicted_audio: torch.Tensor,
        target_audio: torch.Tensor,
        predicted_mel: Optional[torch.Tensor] = None,
        target_mel: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-scale audio reconstruction loss
        
        Args:
            predicted_audio: (B, 1, T) predicted audio waveform
            target_audio: (B, 1, T) target audio waveform
            predicted_mel: Optional predicted mel-spectrogram
            target_mel: Optional target mel-spectrogram
            
        Returns:
            Dictionary with loss components
        """
        losses = {}
        
        # Time-domain losses
        if self.l1_weight > 0:
            losses["l1"] = F.l1_loss(predicted_audio, target_audio) * self.l1_weight
        
        if self.l2_weight > 0:
            losses["l2"] = F.mse_loss(predicted_audio, target_audio) * self.l2_weight
        
        # Spectral losses
        if self.stft_weight > 0:
            stft_loss = 0
            for params in self.stft_params:
                pred_spec = self._compute_spectrogram(predicted_audio, **params)
                target_spec = self._compute_spectrogram(target_audio, **params)
                stft_loss += F.l1_loss(pred_spec, target_spec)
            losses["stft"] = stft_loss / len(self.stft_params) * self.stft_weight
        
        # Mel-spectrogram loss
        if self.mel_weight > 0 and predicted_mel is not None and target_mel is not None:
            losses["mel"] = F.l1_loss(predicted_mel, target_mel) * self.mel_weight
        
        # Total loss
        losses["total"] = sum(losses.values())
        
        return losses
    
    def _compute_spectrogram(
        self,
        audio: torch.Tensor,
        n_fft: int,
        hop_length: int,
        win_length: int,
    ) -> torch.Tensor:
        """Compute magnitude spectrogram"""
        # Remove channel dimension for STFT
        audio = audio.squeeze(1)
        
        # Compute STFT
        spec = torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length, device=audio.device),
            return_complex=True,
        )
        
        # Get magnitude
        magnitude = torch.abs(spec)
        
        # Log scale
        log_magnitude = torch.log(magnitude + 1e-10)
        
        return log_magnitude


class AlignmentLoss(nn.Module):
    """Cross-modal alignment loss for text-audio synchronization"""
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        phoneme_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute alignment loss between modalities
        
        Args:
            text_features: (B, T_text, D) text features
            audio_features: (B, T_audio, D) audio features
            phoneme_features: (B, T_phoneme, D) optional phoneme features
            
        Returns:
            Dictionary with alignment losses
        """
        losses = {}
        
        # Global alignment: pool features across time
        text_global = text_features.mean(dim=1)  # (B, D)
        audio_global = audio_features.mean(dim=1)  # (B, D)
        
        # Normalize
        text_global = F.normalize(text_global, dim=-1)
        audio_global = F.normalize(audio_global, dim=-1)
        
        # Compute similarity
        similarity = torch.matmul(text_global, audio_global.T) / self.temperature
        labels = torch.arange(text_global.shape[0], device=text_global.device)
        
        # Alignment loss
        losses["text_audio_align"] = F.cross_entropy(similarity, labels)
        
        # If phoneme features provided, compute additional alignment
        if phoneme_features is not None:
            phoneme_global = F.normalize(phoneme_features.mean(dim=1), dim=-1)
            
            # Phoneme-audio alignment
            phon_audio_sim = torch.matmul(phoneme_global, audio_global.T) / self.temperature
            losses["phoneme_audio_align"] = F.cross_entropy(phon_audio_sim, labels)
            
            # Phoneme-text alignment
            phon_text_sim = torch.matmul(phoneme_global, text_global.T) / self.temperature
            losses["phoneme_text_align"] = F.cross_entropy(phon_text_sim, labels)
        
        # Total alignment loss
        losses["total"] = sum(losses.values())
        
        return losses


class NLinkLoss(nn.Module):
    """Combined loss function for N-Link training"""
    
    def __init__(
        self,
        contrastive_weight: float = 1.0,
        text_weight: float = 1.0,
        phoneme_weight: float = 0.3,
        audio_weight: float = 0.2,
        alignment_weight: float = 0.1,
    ):
        super().__init__()
        
        # Loss weights
        self.contrastive_weight = contrastive_weight
        self.text_weight = text_weight
        self.phoneme_weight = phoneme_weight
        self.audio_weight = audio_weight
        self.alignment_weight = alignment_weight
        
        # Individual loss functions
        self.contrastive_loss = InfoNCELoss()
        self.clip_loss = CLIPLoss()
        self.ctc_loss = CTCPhonemeLoss()
        self.audio_loss = AudioReconstructionLoss()
        self.alignment_loss = AlignmentLoss()
    
    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        stage: str = "full",
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss based on training stage
        
        Args:
            model_outputs: Dictionary with model predictions
            targets: Dictionary with ground truth values
            stage: Training stage ("contrastive", "brain2text", "full")
            
        Returns:
            Dictionary with all loss components
        """
        losses = {}
        
        if stage == "contrastive":
            # Stage 1: MEG-Mel contrastive learning
            if "meg_features" in model_outputs and "mel_features" in model_outputs:
                losses["contrastive"] = self.contrastive_loss(
                    model_outputs["meg_features"],
                    model_outputs["mel_features"],
                ) * self.contrastive_weight
        
        elif stage == "brain2text":
            # Stage 2: Brain-to-text alignment
            if "brain_features" in model_outputs and "text_features" in targets:
                clip_losses = self.clip_loss(
                    model_outputs["brain_features"],
                    targets["text_features"],
                )
                losses.update({
                    f"clip_{k}": v * self.text_weight 
                    for k, v in clip_losses.items()
                })
        
        elif stage == "full":
            # Stage 3: Full multi-task training
            
            # Text generation loss
            if "text_logits" in model_outputs and "text_ids" in targets:
                text_loss = F.cross_entropy(
                    model_outputs["text_logits"].reshape(-1, model_outputs["text_logits"].size(-1)),
                    targets["text_ids"].reshape(-1),
                    ignore_index=-100,
                )
                losses["text"] = text_loss * self.text_weight
            
            # Phoneme prediction loss
            if "phoneme_logits" in model_outputs and "phoneme_ids" in targets:
                phoneme_loss = self.ctc_loss(
                    F.log_softmax(model_outputs["phoneme_logits"], dim=-1).transpose(0, 1),
                    targets["phoneme_ids"],
                    targets["phoneme_lengths"],
                    targets["phoneme_target_lengths"],
                )
                losses["phoneme"] = phoneme_loss * self.phoneme_weight
            
            # Audio reconstruction loss
            if "audio" in model_outputs and "audio" in targets:
                audio_losses = self.audio_loss(
                    model_outputs["audio"],
                    targets["audio"],
                    model_outputs.get("mel"),
                    targets.get("mel"),
                )
                losses.update({
                    f"audio_{k}": v * self.audio_weight 
                    for k, v in audio_losses.items()
                })
            
            # Cross-modal alignment loss
            if self.alignment_weight > 0:
                align_losses = self.alignment_loss(
                    model_outputs.get("text_features"),
                    model_outputs.get("audio_features"),
                    model_outputs.get("phoneme_features"),
                )
                losses.update({
                    f"align_{k}": v * self.alignment_weight 
                    for k, v in align_losses.items()
                })
        
        # Compute total loss
        losses["total"] = sum(v for k, v in losses.items() if "total" not in k)
        
        return losses