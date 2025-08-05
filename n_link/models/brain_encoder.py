import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple
from einops import rearrange


class ConvBlock(nn.Module):
    """Convolutional block with residual connection"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        groups: int = 8,
    ):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activation = nn.GELU()
        
        # Residual connection
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x + residual


class SpatialAttention(nn.Module):
    """MEG channel-wise spatial attention"""
    
    def __init__(self, num_channels: int = 208):
        super().__init__()
        # Use Linear layers for channel attention instead of Conv1d
        self.attention = nn.Sequential(
            nn.Linear(num_channels, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, num_channels),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        # Use global average pooling to get channel statistics
        x_pooled = x.mean(dim=2)  # (B, C) - pool over time
        attention_weights = self.attention(x_pooled)  # (B, C)
        attention_weights = attention_weights.unsqueeze(-1)  # (B, C, 1)
        return x * attention_weights  # Broadcasting will handle the multiplication


class FrequencyEncoder(nn.Module):
    """Multi-scale STFT-based frequency encoder"""
    
    def __init__(
        self,
        num_channels: int = 208,
        sampling_rate: int = 1000,
        freq_windows: List[int] = [32, 64, 128],
        embed_dim: int = 64,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.sampling_rate = sampling_rate
        self.freq_windows = freq_windows
        self.embed_dim = embed_dim
        
        # Projections for each frequency scale
        self.freq_projections = nn.ModuleList([
            nn.Linear((win // 2 + 1) * num_channels, embed_dim)
            for win in freq_windows
        ])
        
        # Cross-attention for frequency fusion
        self.freq_attention = nn.MultiheadAttention(
            embed_dim=embed_dim * len(freq_windows),
            num_heads=8,
            batch_first=True,
        )
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim * len(freq_windows), 192)
    
    def compute_stft(self, x: torch.Tensor, window_size: int) -> torch.Tensor:
        """Compute STFT for a given window size"""
        # x: (B, C, T)
        B, C, T = x.shape
        
        # Apply STFT per channel
        stft_result = torch.stft(
            x.reshape(B * C, T),
            n_fft=window_size,
            hop_length=window_size // 4,
            window=torch.hann_window(window_size, device=x.device),
            return_complex=True,
        )  # (B*C, freq_bins, time_frames)
        
        # Get magnitude
        magnitude = torch.abs(stft_result)
        
        # Reshape back
        freq_bins = magnitude.shape[1]
        time_frames = magnitude.shape[2]
        magnitude = magnitude.reshape(B, C, freq_bins, time_frames)
        
        # Flatten channel and frequency dimensions
        magnitude = magnitude.permute(0, 3, 1, 2).reshape(B, time_frames, -1)
        
        return magnitude
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        freq_features = []
        
        # Extract features at multiple frequency scales
        target_len = None
        for i, window_size in enumerate(self.freq_windows):
            stft_feat = self.compute_stft(x, window_size)  # (B, T', C*freq_bins)
            proj_feat = self.freq_projections[i](stft_feat)  # (B, T', embed_dim)
            
            # Interpolate to common temporal dimension (use first scale as reference)
            if target_len is None:
                target_len = proj_feat.shape[1]
            elif proj_feat.shape[1] != target_len:
                # Interpolate to match target length
                proj_feat = F.interpolate(
                    proj_feat.transpose(1, 2),  # (B, embed_dim, T')
                    size=target_len,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)  # (B, T', embed_dim)
            
            freq_features.append(proj_feat)
        
        # Concatenate all frequency features
        multi_scale_features = torch.cat(freq_features, dim=-1)  # (B, T', embed_dim*3)
        
        # Self-attention across time
        attended_features, _ = self.freq_attention(
            multi_scale_features,
            multi_scale_features,
            multi_scale_features,
        )
        
        # Project to output dimension
        output = self.output_proj(attended_features)  # (B, T', 192)
        
        return output


class MEGBrainEncoder(nn.Module):
    """
    MEG Brain Encoder combining temporal and frequency pathways
    Based on Defossez et al. architecture adapted for MEG
    """
    
    def __init__(
        self,
        meg_channels: int = 208,
        sampling_rate: int = 1000,
        num_subjects: int = 27,
        output_dim: int = 384,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.meg_channels = meg_channels
        self.sampling_rate = sampling_rate
        self.output_dim = output_dim
        
        # Spatial attention for MEG channels
        self.spatial_attention = SpatialAttention(meg_channels)
        
        # Subject embedding
        self.subject_embedding = nn.Embedding(num_subjects, meg_channels)
        
        # Temporal encoder (progressive dilation)
        self.temporal_encoder = nn.ModuleList([
            ConvBlock(meg_channels, 320, kernel_size=3, dilation=1),
            ConvBlock(320, 320, kernel_size=3, dilation=2),
            ConvBlock(320, 320, kernel_size=3, dilation=4),
            ConvBlock(320, 256, kernel_size=3, dilation=8),
        ])
        
        # Frequency encoder
        self.frequency_encoder = FrequencyEncoder(
            num_channels=meg_channels,
            sampling_rate=sampling_rate,
            freq_windows=[32, 64, 128],
            embed_dim=64,
        )
        
        # Temporal pooling
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(256 + 192, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim),
        )
        
        # Layer normalization for output
        self.output_norm = nn.LayerNorm(output_dim)
    
    def forward(
        self,
        meg_signal: torch.Tensor,
        subject_id: Optional[torch.Tensor] = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through MEG brain encoder
        
        Args:
            meg_signal: (B, C, T) MEG signals
            subject_id: (B,) subject IDs for embedding
            return_sequence: If True, return sequence features instead of pooled
            
        Returns:
            features: (B, D) pooled features or (B, T', D) sequence features
        """
        B, C, T = meg_signal.shape
        
        # Apply spatial attention
        x = self.spatial_attention(meg_signal)
        
        # Add subject embedding if provided
        if subject_id is not None:
            subject_emb = self.subject_embedding(subject_id)  # (B, C)
            x = x + subject_emb.unsqueeze(-1)
        
        # Temporal pathway
        temporal_features = x
        for conv_block in self.temporal_encoder:
            temporal_features = conv_block(temporal_features)
        # temporal_features: (B, 256, T')
        
        # Frequency pathway
        freq_features = self.frequency_encoder(x)  # (B, T'', 192)
        
        if return_sequence:
            # Align temporal dimensions
            temp_seq = temporal_features.permute(0, 2, 1)  # (B, T', 256)
            
            # Interpolate frequency features to match temporal dimension
            T_temp = temp_seq.shape[1]
            T_freq = freq_features.shape[1]
            
            if T_temp != T_freq:
                freq_features = F.interpolate(
                    freq_features.permute(0, 2, 1),
                    size=T_temp,
                    mode='linear',
                    align_corners=False,
                ).permute(0, 2, 1)
            
            # Concatenate and fuse
            combined = torch.cat([temp_seq, freq_features], dim=-1)  # (B, T', 448)
            output = self.fusion(combined)  # (B, T', output_dim)
            output = self.output_norm(output)
            
            return output
        
        else:
            # Pool temporal features
            temporal_pooled = self.temporal_pool(temporal_features).squeeze(-1)  # (B, 256)
            
            # Pool frequency features
            freq_pooled = freq_features.mean(dim=1)  # (B, 192)
            
            # Concatenate and fuse
            combined = torch.cat([temporal_pooled, freq_pooled], dim=-1)  # (B, 448)
            output = self.fusion(combined)  # (B, output_dim)
            output = self.output_norm(output)
            
            return output
    
    def extract_multilevel_features(
        self,
        meg_signal: torch.Tensor,
        subject_id: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Extract features at multiple levels for analysis
        
        Args:
            meg_signal: (B, C, T) MEG signals
            subject_id: (B,) subject IDs
            
        Returns:
            Dictionary with features at different stages
        """
        B, C, T = meg_signal.shape
        features = {}
        
        # Spatial attention
        x = self.spatial_attention(meg_signal)
        features['spatial_attended'] = x
        
        # Add subject embedding
        if subject_id is not None:
            subject_emb = self.subject_embedding(subject_id)
            x = x + subject_emb.unsqueeze(-1)
        
        # Temporal features at each layer
        temporal_features = x
        for i, conv_block in enumerate(self.temporal_encoder):
            temporal_features = conv_block(temporal_features)
            features[f'temporal_layer_{i}'] = temporal_features
        
        # Frequency features
        freq_features = self.frequency_encoder(x)
        features['frequency'] = freq_features
        
        # Final fused features
        final_features = self.forward(meg_signal, subject_id, return_sequence=True)
        features['final'] = final_features
        
        return features