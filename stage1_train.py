#!/usr/bin/env python3
"""
Stage 1 Training Script: MEG-Mel Alignment
This script trains the MEG-Mel alignment model using contrastive learning
"""

import argparse
import torch
import wandb
from pathlib import Path
import json
from datetime import datetime

from n_link.data import create_dataloaders
from n_link.training import NLinkTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: MEG-Mel Alignment Training")
    
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
        "--mel_bins",
        type=int,
        default=80,
        help="Number of mel-spectrogram bins",
    )
    
    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--contrastive_weight",
        type=float,
        default=1.0,
        help="Weight for contrastive loss",
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
        default="n-link-stage1",
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
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create experiment name
    if args.experiment_name is None:
        args.experiment_name = f"stage1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
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
    
    print(f"Stage 1 Training Configuration:")
    print(f"  Experiment: {args.experiment_name}")
    print(f"  Data root: {args.data_root}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Device: {args.device}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            cache_dir=args.cache_dir,
            meg_channels=args.meg_channels,
            sampling_rate=args.sampling_rate,
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
        "mel_bins": args.mel_bins,
        "num_subjects": 27,  # MASC-MEG has 27 subjects
        
        # Training parameters
        "learning_rate_stage1": args.learning_rate,
        "contrastive_weight": args.contrastive_weight,
        
        # Other parameters
        "device": args.device,
    }
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = NLinkTrainer(
        config=trainer_config,
        device=args.device,
        use_wandb=args.use_wandb,
        checkpoint_dir=str(checkpoint_dir),
    )
    
    # Train Stage 1
    print("\nStarting Stage 1 training...")
    try:
        trainer.train_stage1(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
        )
        print("\nStage 1 training completed successfully!")
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