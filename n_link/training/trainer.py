import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Tuple
import wandb
from tqdm import tqdm
import os
import json
from pathlib import Path

from ..models import (
    MEGBrainEncoder,
    MEGMelAligner,
    MEGToLLaVAAdapter,
    MEGLLaVAInterface,
    MultiOutputDecoder,
)
from .losses import NLinkLoss


class NLinkTrainer:
    """Multi-stage trainer for N-Link model"""
    
    def __init__(
        self,
        config: Dict,
        device: str = "cuda",
        use_wandb: bool = True,
        checkpoint_dir: str = "./checkpoints",
    ):
        self.config = config
        self.device = device
        self.use_wandb = use_wandb
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self._build_models()
        
        # Initialize loss function
        self.criterion = NLinkLoss(
            contrastive_weight=config.get("contrastive_weight", 1.0),
            text_weight=config.get("text_weight", 1.0),
            phoneme_weight=config.get("phoneme_weight", 0.3),
            audio_weight=config.get("audio_weight", 0.2),
            alignment_weight=config.get("alignment_weight", 0.1),
        )
        
        # Initialize optimizers
        self._build_optimizers()
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Initialize wandb if requested
        if self.use_wandb:
            wandb.init(project="n-link", config=config)
    
    def _build_models(self):
        """Initialize all model components"""
        config = self.config
        
        # MEG-Mel Aligner
        self.meg_mel_aligner = MEGMelAligner(
            meg_channels=config.get("meg_channels", 208),
            sampling_rate=config.get("sampling_rate", 1000),
            mel_bins=config.get("mel_bins", 80),
        ).to(self.device)
        
        # Brain Encoder
        self.brain_encoder = MEGBrainEncoder(
            meg_channels=config.get("meg_channels", 208),
            sampling_rate=config.get("sampling_rate", 1000),
            num_subjects=config.get("num_subjects", 27),
            output_dim=config.get("brain_encoder_dim", 384),
        ).to(self.device)
        
        # MEG to LLaVA Adapter
        self.meg_llava_adapter = MEGToLLaVAAdapter(
            meg_feature_dim=config.get("brain_encoder_dim", 384),
            llava_visual_dim=config.get("llava_visual_dim", 576),
            num_visual_tokens=config.get("num_visual_tokens", 256),
        ).to(self.device)
        
        # LLaVA Interface
        self.llava_interface = MEGLLaVAInterface(
            brain_encoder=self.brain_encoder,
            adapter=self.meg_llava_adapter,
            llava_model_name=config.get("llava_model_name", "llava-hf/llava-1.5-7b-hf"),
            freeze_llava=config.get("freeze_llava", True),
            use_lora=config.get("use_lora", True),
        ).to(self.device)
        
        # Multi-output Decoder
        self.multi_decoder = MultiOutputDecoder(
            llava_hidden_dim=config.get("llava_hidden_dim", 4096),
            shared_dim=config.get("decoder_shared_dim", 768),
            num_phonemes=config.get("num_phonemes", 70),
        ).to(self.device)
    
    def _build_optimizers(self):
        """Initialize optimizers for different training stages"""
        config = self.config
        
        # Stage 1: MEG-Mel alignment
        self.optimizer_stage1 = AdamW(
            self.meg_mel_aligner.parameters(),
            lr=config.get("lr_stage1", 1e-3),
            weight_decay=config.get("weight_decay", 0.01),
        )
        
        # Stage 2: Brain-to-text
        stage2_params = list(self.brain_encoder.parameters()) + \
                       list(self.meg_llava_adapter.parameters())
        self.optimizer_stage2 = AdamW(
            stage2_params,
            lr=config.get("lr_stage2", 5e-4),
            weight_decay=config.get("weight_decay", 0.01),
        )
        
        # Stage 3: Full model
        stage3_params = []
        for model in [self.brain_encoder, self.meg_llava_adapter, 
                     self.multi_decoder, self.llava_interface]:
            stage3_params.extend([p for p in model.parameters() if p.requires_grad])
        
        self.optimizer_stage3 = AdamW(
            stage3_params,
            lr=config.get("lr_stage3", 1e-4),
            weight_decay=config.get("weight_decay", 0.01),
        )
        
        # Learning rate schedulers
        self.scheduler_stage1 = CosineAnnealingWarmRestarts(
            self.optimizer_stage1, T_0=10, T_mult=2
        )
        self.scheduler_stage2 = CosineAnnealingWarmRestarts(
            self.optimizer_stage2, T_0=10, T_mult=2
        )
        self.scheduler_stage3 = CosineAnnealingWarmRestarts(
            self.optimizer_stage3, T_0=10, T_mult=2
        )
    
    def train_stage1(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
    ):
        """Stage 1: MEG-Mel contrastive pre-training"""
        print("Starting Stage 1: MEG-Mel Alignment")
        
        self.meg_mel_aligner.train()
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self._train_epoch_stage1(train_loader, epoch)
            
            # Validation
            val_loss = self._validate_stage1(val_loader)
            
            # Learning rate scheduling
            self.scheduler_stage1.step()
            
            # Logging
            if self.use_wandb:
                wandb.log({
                    "stage1/train_loss": train_loss,
                    "stage1/val_loss": val_loss,
                    "stage1/lr": self.optimizer_stage1.param_groups[0]['lr'],
                    "epoch": epoch,
                })
            
            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint("stage1", epoch, val_loss)
            
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    def _train_epoch_stage1(self, train_loader: DataLoader, epoch: int) -> float:
        """Train one epoch of stage 1"""
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Stage 1 - Epoch {epoch}")
        for batch in pbar:
            meg_signal = batch["meg_signal"].to(self.device)
            audio = batch["audio"].to(self.device)
            
            self.optimizer_stage1.zero_grad()
            
            with autocast():
                outputs = self.meg_mel_aligner(meg_signal, audio=audio)
                loss = outputs["loss"]
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer_stage1)
            self.scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({"loss": loss.item()})
        
        return total_loss / num_batches
    
    def _validate_stage1(self, val_loader: DataLoader) -> float:
        """Validate stage 1"""
        self.meg_mel_aligner.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating Stage 1"):
                meg_signal = batch["meg_signal"].to(self.device)
                audio = batch["audio"].to(self.device)
                
                outputs = self.meg_mel_aligner(meg_signal, audio=audio)
                loss = outputs["loss"]
                
                total_loss += loss.item()
                num_batches += 1
        
        self.meg_mel_aligner.train()
        return total_loss / num_batches
    
    def train_stage2(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 30,
    ):
        """Stage 2: Brain-to-text training"""
        print("Starting Stage 2: Brain-to-Text Alignment")
        
        # Load best stage 1 checkpoint
        self._load_checkpoint("stage1")
        
        self.brain_encoder.train()
        self.meg_llava_adapter.train()
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self._train_epoch_stage2(train_loader, epoch)
            
            # Validation
            val_loss = self._validate_stage2(val_loader)
            
            # Learning rate scheduling
            self.scheduler_stage2.step()
            
            # Logging
            if self.use_wandb:
                wandb.log({
                    "stage2/train_loss": train_loss,
                    "stage2/val_loss": val_loss,
                    "stage2/lr": self.optimizer_stage2.param_groups[0]['lr'],
                    "epoch": epoch,
                })
            
            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint("stage2", epoch, val_loss)
            
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    def _train_epoch_stage2(self, train_loader: DataLoader, epoch: int) -> float:
        """Train one epoch of stage 2"""
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Stage 2 - Epoch {epoch}")
        for batch in pbar:
            meg_signal = batch["meg_signal"].to(self.device)
            subject_id = batch.get("subject_id", None)
            if subject_id is not None:
                subject_id = subject_id.to(self.device)
            
            self.optimizer_stage2.zero_grad()
            
            with autocast():
                # Get brain features
                brain_features = self.brain_encoder(meg_signal, subject_id)
                
                # Get text features from targets
                text_features = batch["text_features"].to(self.device)
                
                # Compute loss
                losses = self.criterion(
                    {"brain_features": brain_features},
                    {"text_features": text_features},
                    stage="brain2text"
                )
                loss = losses["total"]
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer_stage2)
            self.scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({"loss": loss.item()})
        
        return total_loss / num_batches
    
    def _validate_stage2(self, val_loader: DataLoader) -> float:
        """Validate stage 2"""
        self.brain_encoder.eval()
        self.meg_llava_adapter.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating Stage 2"):
                meg_signal = batch["meg_signal"].to(self.device)
                subject_id = batch.get("subject_id", None)
                if subject_id is not None:
                    subject_id = subject_id.to(self.device)
                
                # Get brain features
                brain_features = self.brain_encoder(meg_signal, subject_id)
                
                # Get text features from targets
                text_features = batch["text_features"].to(self.device)
                
                # Compute loss
                losses = self.criterion(
                    {"brain_features": brain_features},
                    {"text_features": text_features},
                    stage="brain2text"
                )
                loss = losses["total"]
                
                total_loss += loss.item()
                num_batches += 1
        
        self.brain_encoder.train()
        self.meg_llava_adapter.train()
        return total_loss / num_batches
    
    def train_stage3(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
    ):
        """Stage 3: End-to-end fine-tuning"""
        print("Starting Stage 3: End-to-End Fine-tuning")
        
        # Load best stage 2 checkpoint
        self._load_checkpoint("stage2")
        
        # Set training mode
        self.llava_interface.prepare_for_training()
        self.multi_decoder.train()
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            train_losses = self._train_epoch_stage3(train_loader, epoch)
            
            # Validation
            val_losses = self._validate_stage3(val_loader)
            
            # Learning rate scheduling
            self.scheduler_stage3.step()
            
            # Logging
            if self.use_wandb:
                log_dict = {
                    f"stage3/train_{k}": v for k, v in train_losses.items()
                }
                log_dict.update({
                    f"stage3/val_{k}": v for k, v in val_losses.items()
                })
                log_dict["stage3/lr"] = self.optimizer_stage3.param_groups[0]['lr']
                log_dict["epoch"] = epoch
                wandb.log(log_dict)
            
            # Save checkpoint
            val_loss = val_losses.get("total", float('inf'))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint("stage3", epoch, val_loss)
            
            print(f"Epoch {epoch}: Train Loss: {train_losses['total']:.4f}, "
                  f"Val Loss: {val_losses['total']:.4f}")
    
    def _train_epoch_stage3(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train one epoch of stage 3"""
        loss_accumulator = {}
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Stage 3 - Epoch {epoch}")
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            self.optimizer_stage3.zero_grad()
            
            with autocast():
                # Forward pass through full model
                model_outputs = self._forward_stage3(batch)
                
                # Prepare targets
                targets = self._prepare_targets(batch)
                
                # Compute losses
                losses = self.criterion(model_outputs, targets, stage="full")
                loss = losses["total"]
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer_stage3)
            self.scaler.update()
            
            # Accumulate losses
            for k, v in losses.items():
                if k not in loss_accumulator:
                    loss_accumulator[k] = 0
                loss_accumulator[k] += v.item()
            
            num_batches += 1
            pbar.set_postfix({"loss": loss.item()})
        
        # Average losses
        return {k: v / num_batches for k, v in loss_accumulator.items()}
    
    def _forward_stage3(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Forward pass for stage 3"""
        meg_signal = batch["meg_signal"]
        subject_id = batch.get("subject_id")
        
        # Get LLaVA interface outputs
        llava_outputs = self.llava_interface(
            meg_signal,
            subject_id=subject_id,
        )
        
        # Generate multi-modal outputs
        # Note: In real implementation, we'd need actual LLaVA hidden states
        # This is a simplified version
        dummy_hidden_states = torch.randn(
            meg_signal.shape[0], 
            100,  # sequence length
            self.config.get("llava_hidden_dim", 4096),
            device=self.device
        )
        
        # Only request modalities with non-zero weights
        target_modalities = []
        if self.config.get("text_weight", 1.0) > 0:
            target_modalities.append("text")
        if self.config.get("phoneme_weight", 0.3) > 0:
            target_modalities.append("phoneme")
        if self.config.get("audio_weight", 0.2) > 0:
            target_modalities.append("audio")
            
        decoder_outputs = self.multi_decoder(
            dummy_hidden_states,
            llava_outputs=llava_outputs,
            target_modalities=target_modalities,
        )
        
        # Combine outputs
        model_outputs = {
            **llava_outputs,
            **decoder_outputs,
        }
        
        return model_outputs
    
    def _prepare_targets(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Prepare target tensors for loss computation"""
        targets = {}
        
        if "text_ids" in batch:
            targets["text_ids"] = batch["text_ids"]
        
        if "phoneme_ids" in batch:
            targets["phoneme_ids"] = batch["phoneme_ids"]
            targets["phoneme_lengths"] = batch["phoneme_lengths"]
            targets["phoneme_target_lengths"] = batch["phoneme_target_lengths"]
        
        if "audio" in batch:
            targets["audio"] = batch["audio"]
        
        if "mel" in batch:
            targets["mel"] = batch["mel"]
        
        return targets
    
    def _validate_stage3(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate stage 3"""
        # Set eval mode
        self.brain_encoder.eval()
        self.meg_llava_adapter.eval()
        self.multi_decoder.eval()
        
        loss_accumulator = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating Stage 3"):
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                model_outputs = self._forward_stage3(batch)
                targets = self._prepare_targets(batch)
                
                # Compute losses
                losses = self.criterion(model_outputs, targets, stage="full")
                
                # Accumulate losses
                for k, v in losses.items():
                    if k not in loss_accumulator:
                        loss_accumulator[k] = 0
                    loss_accumulator[k] += v.item()
                
                num_batches += 1
        
        # Set back to train mode
        self.llava_interface.prepare_for_training()
        self.multi_decoder.train()
        
        # Average losses
        return {k: v / num_batches for k, v in loss_accumulator.items()}
    
    def _save_checkpoint(self, stage: str, epoch: int, val_loss: float):
        """Save model checkpoint"""
        checkpoint = {
            "stage": stage,
            "epoch": epoch,
            "val_loss": val_loss,
            "config": self.config,
        }
        
        if stage == "stage1":
            checkpoint["meg_mel_aligner"] = self.meg_mel_aligner.state_dict()
            checkpoint["optimizer"] = self.optimizer_stage1.state_dict()
        elif stage == "stage2":
            checkpoint["meg_mel_aligner"] = self.meg_mel_aligner.state_dict()
            checkpoint["brain_encoder"] = self.brain_encoder.state_dict()
            checkpoint["meg_llava_adapter"] = self.meg_llava_adapter.state_dict()
            checkpoint["optimizer"] = self.optimizer_stage2.state_dict()
        elif stage == "stage3":
            checkpoint["meg_mel_aligner"] = self.meg_mel_aligner.state_dict()
            checkpoint["brain_encoder"] = self.brain_encoder.state_dict()
            checkpoint["meg_llava_adapter"] = self.meg_llava_adapter.state_dict()
            checkpoint["multi_decoder"] = self.multi_decoder.state_dict()
            checkpoint["optimizer"] = self.optimizer_stage3.state_dict()
        
        path = self.checkpoint_dir / f"{stage}_best.pt"
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def _load_checkpoint(self, stage: str):
        """Load model checkpoint"""
        path = self.checkpoint_dir / f"{stage}_best.pt"
        if not path.exists():
            print(f"No checkpoint found for {stage}")
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        
        if "meg_mel_aligner" in checkpoint:
            self.meg_mel_aligner.load_state_dict(checkpoint["meg_mel_aligner"])
        
        if "brain_encoder" in checkpoint:
            self.brain_encoder.load_state_dict(checkpoint["brain_encoder"])
        
        if "meg_llava_adapter" in checkpoint:
            self.meg_llava_adapter.load_state_dict(checkpoint["meg_llava_adapter"])
        
        if "multi_decoder" in checkpoint:
            self.multi_decoder.load_state_dict(checkpoint["multi_decoder"])
        
        print(f"Loaded checkpoint from {path}")