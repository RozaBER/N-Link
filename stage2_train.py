#!/usr/bin/env python3
"""
Stage 2 Training Script: Brain-to-Text Alignment
This script trains the brain encoder and MEG-to-LLaVA adapter
"""

import argparse
import torch
import wandb
from pathlib import Path
import json
from datetime import datetime

from n_link.data import create_dataloaders
from n_link.training import NLinkTrainer
from torch.utils.data import Dataset, DataLoader


class TextFeatureWrapper(Dataset):
    """Wrapper that adds text features to batches for Stage 2 training"""
    
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        batch = self.dataset[idx]
        
        # Stage 2 needs text features but dataset provides raw text
        # Create proxy text features from MEG signal
        meg_signal = batch['meg_signal']  # (C, T)
        
        # Simple pooling to create a feature vector
        text_features = meg_signal.mean(dim=1)  # (C,) = (208,)
        
        # Project to 384 dimensions to match brain encoder output
        if text_features.shape[0] < 384:
            text_features = torch.nn.functional.pad(text_features, (0, 384 - text_features.shape[0]))
        else:
            text_features = text_features[:384]
        
        batch['text_features'] = text_features
        return batch


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: Brain-to-Text Alignment Training")
    
    # Data arguments
    parser.add_argument(
        "--data_root",
        type=str,
        default="/data/zshao/masc_meg",
        help="Path to MASC-MEG dataset",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Directory for caching preprocessed data",
    )
    
    # Model arguments
    parser.add_argument(
        "--meg_channels",
        type=int,
        default=208,
        help="Number of MEG channels",
    )
    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=1000,
        help="MEG sampling rate (Hz)",
    )
    parser.add_argument(
        "--brain_encoder_dim",
        type=int,
        default=384,
        help="Brain encoder output dimension",
    )
    parser.add_argument(
        "--llava_model_name",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="LLaVA model name",
    )
    parser.add_argument(
        "--llava_visual_dim",
        type=int,
        default=576,
        help="LLaVA visual dimension",
    )
    parser.add_argument(
        "--num_visual_tokens",
        type=int,
        default=256,
        help="Number of visual tokens for LLaVA",
    )
    
    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training batch size (smaller for Stage 2 due to LLaVA)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--text_weight",
        type=float,
        default=1.0,
        help="Weight for text generation loss",
    )
    parser.add_argument(
        "--clip_weight",
        type=float,
        default=0.1,
        help="Weight for CLIP-style contrastive loss",
    )
    
    # Stage 1 checkpoint
    parser.add_argument(
        "--stage1_checkpoint",
        type=str,
        required=True,
        help="Path to Stage 1 checkpoint directory",
    )
    
    # System arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases for logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="n-link-stage2",
        help="W&B project name",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment name (defaults to timestamp)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--freeze_llava",
        type=bool,
        default=True,
        help="Freeze LLaVA backbone",
    )
    parser.add_argument(
        "--use_lora",
        type=bool,
        default=True,
        help="Use LoRA for LLaVA fine-tuning",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create experiment name
    if args.experiment_name is None:
        args.experiment_name = f"stage2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize W&B if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.experiment_name,
            config=vars(args),
        )
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir) / args.experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = checkpoint_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"Stage 2 Training Configuration:")
    print(f"  Experiment: {args.experiment_name}")
    print(f"  Data root: {args.data_root}")
    print(f"  Stage 1 checkpoint: {args.stage1_checkpoint}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Device: {args.device}")
    print(f"  LLaVA model: {args.llava_model_name}")
    print(f"  Freeze LLaVA: {args.freeze_llava}")
    print(f"  Use LoRA: {args.use_lora}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    try:
        from n_link.data.masc_meg_dataset import MASCMEGDataset
        
        # Create datasets
        train_dataset = MASCMEGDataset(
            data_root=args.data_root,
            split='train',
            cache_dir=args.cache_dir,
            meg_channels=args.meg_channels,
            sampling_rate=args.sampling_rate,
        )
        
        val_dataset = MASCMEGDataset(
            data_root=args.data_root,
            split='val',
            cache_dir=args.cache_dir,
            meg_channels=args.meg_channels,
            sampling_rate=args.sampling_rate,
        )
        
        test_dataset = MASCMEGDataset(
            data_root=args.data_root,
            split='test',
            cache_dir=args.cache_dir,
            meg_channels=args.meg_channels,
            sampling_rate=args.sampling_rate,
        )
        
        # Wrap datasets to add text features
        train_dataset = TextFeatureWrapper(train_dataset)
        val_dataset = TextFeatureWrapper(val_dataset)
        test_dataset = TextFeatureWrapper(test_dataset)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        print(f"  Train samples: {len(train_loader.dataset)}")
        print(f"  Val samples: {len(val_loader.dataset)}")
        print(f"  Test samples: {len(test_loader.dataset)}")
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        raise
    
    # Create trainer configuration
    trainer_config = {
        # Model parameters
        "meg_channels": args.meg_channels,
        "sampling_rate": args.sampling_rate,
        "num_subjects": 28,  # MASC has subjects 1-27, need 28 for indexing
        "brain_encoder_dim": args.brain_encoder_dim,
        "llava_model_name": args.llava_model_name,
        "llava_visual_dim": args.llava_visual_dim,
        "num_visual_tokens": args.num_visual_tokens,
        "freeze_llava": args.freeze_llava,
        "use_lora": args.use_lora,
        
        # Training parameters
        "learning_rate_stage2": args.learning_rate,
        "text_weight": args.text_weight,
        "clip_weight": args.clip_weight,
        
        # Other parameters
        "device": args.device,
        "stage1_checkpoint": args.stage1_checkpoint,
    }
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = NLinkTrainer(
        config=trainer_config,
        device=args.device,
        use_wandb=args.use_wandb,
        checkpoint_dir=str(checkpoint_dir),
    )
    
    # Load Stage 1 checkpoint
    print(f"\nLoading Stage 1 checkpoint from {args.stage1_checkpoint}...")
    try:
        # Load checkpoint manually
        checkpoint = torch.load(args.stage1_checkpoint, map_location=args.device)
        
        # Load MEG-Mel aligner weights if available
        if "meg_mel_aligner" in checkpoint:
            trainer.meg_mel_aligner.load_state_dict(checkpoint["meg_mel_aligner"])
            print("Loaded MEG-Mel aligner weights")
        
        # Load brain encoder weights if available
        if "brain_encoder" in checkpoint:
            trainer.brain_encoder.load_state_dict(checkpoint["brain_encoder"])
            print("Loaded brain encoder weights")
            
        print("Stage 1 checkpoint loaded successfully")
    except Exception as e:
        print(f"Error loading Stage 1 checkpoint: {e}")
        raise
    
    # Train Stage 2
    print("\nStarting Stage 2 training...")
    try:
        trainer.train_stage2(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
        )
        print("\nStage 2 training completed successfully!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise
    
    # Close wandb
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()