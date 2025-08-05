#!/usr/bin/env python3
"""
N-Link Training Script
Multi-stage training for MEG-to-Speech/Text model
"""

import argparse
import json
import os
from pathlib import Path
import torch
import wandb
from datetime import datetime

from n_link.data import create_dataloaders, TextFeatureExtractor
from n_link.training import NLinkTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train N-Link MEG-to-Speech/Text Model")
    
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
        "--freeze_llava",
        action="store_true",
        help="Freeze LLaVA weights",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA for LLaVA fine-tuning",
    )
    
    # Training arguments
    parser.add_argument(
        "--stage",
        type=str,
        choices=["all", "stage1", "stage2", "stage3"],
        default="all",
        help="Training stage(s) to run",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--epochs_stage1",
        type=int,
        default=50,
        help="Number of epochs for stage 1",
    )
    parser.add_argument(
        "--epochs_stage2",
        type=int,
        default=30,
        help="Number of epochs for stage 2",
    )
    parser.add_argument(
        "--epochs_stage3",
        type=int,
        default=50,
        help="Number of epochs for stage 3",
    )
    parser.add_argument(
        "--lr_stage1",
        type=float,
        default=1e-3,
        help="Learning rate for stage 1",
    )
    parser.add_argument(
        "--lr_stage2",
        type=float,
        default=5e-4,
        help="Learning rate for stage 2",
    )
    parser.add_argument(
        "--lr_stage3",
        type=float,
        default=1e-4,
        help="Learning rate for stage 3",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    
    # Loss weights
    parser.add_argument(
        "--contrastive_weight",
        type=float,
        default=1.0,
        help="Weight for contrastive loss",
    )
    parser.add_argument(
        "--text_weight",
        type=float,
        default=1.0,
        help="Weight for text generation loss",
    )
    parser.add_argument(
        "--phoneme_weight",
        type=float,
        default=0.3,
        help="Weight for phoneme prediction loss",
    )
    parser.add_argument(
        "--audio_weight",
        type=float,
        default=0.2,
        help="Weight for audio reconstruction loss",
    )
    parser.add_argument(
        "--alignment_weight",
        type=float,
        default=0.1,
        help="Weight for cross-modal alignment loss",
    )
    
    # Other arguments
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases for logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="n-link",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    return parser.parse_args()


def setup_wandb(args):
    """Initialize Weights & Biases"""
    if args.use_wandb:
        run_name = args.wandb_run_name or f"n-link_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
        )
        
        # Log code
        wandb.run.log_code(".")


def prepare_text_features(args, train_loader, val_loader, test_loader):
    """Pre-extract text features for stage 2 training"""
    print("Extracting text features for stage 2 training...")
    
    # Initialize text feature extractor
    text_extractor = TextFeatureExtractor(
        model_name="openai/clip-vit-base-patch32",
        device=args.device,
        use_clip=True,
    )
    
    # Cache directory for text features
    text_feature_dir = Path(args.cache_dir) / "text_features"
    text_feature_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each dataset
    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        feature_file = text_feature_dir / f"{split_name}_text_features.pt"
        
        if feature_file.exists():
            print(f"Text features for {split_name} already exist, skipping...")
            continue
        
        print(f"Extracting text features for {split_name} split...")
        
        all_features = []
        all_indices = []
        
        for batch_idx, batch in enumerate(loader):
            if 'text' in batch:
                # Extract features
                text_features = text_extractor.extract_features(batch['text'])
                
                # Store features and indices
                all_features.append(text_features.cpu())
                
                # Calculate global indices
                start_idx = batch_idx * loader.batch_size
                end_idx = start_idx + len(batch['text'])
                all_indices.extend(range(start_idx, end_idx))
        
        if all_features:
            # Save features
            features_dict = {
                'features': torch.cat(all_features, dim=0),
                'indices': all_indices,
            }
            torch.save(features_dict, feature_file)
            print(f"Saved {len(all_indices)} text features to {feature_file}")


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Setup W&B
    setup_wandb(args)
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
        meg_channels=args.meg_channels,
        sampling_rate=args.sampling_rate,
    )
    
    # Prepare text features for stage 2
    if args.stage in ["all", "stage2"]:
        prepare_text_features(args, train_loader, val_loader, test_loader)
    
    # Create trainer configuration
    config = {
        # Model parameters
        'meg_channels': args.meg_channels,
        'sampling_rate': args.sampling_rate,
        'num_subjects': 27,  # MASC-MEG has 27 subjects
        'brain_encoder_dim': args.brain_encoder_dim,
        'llava_model_name': args.llava_model_name,
        'freeze_llava': args.freeze_llava,
        'use_lora': args.use_lora,
        'llava_visual_dim': 576,
        'llava_hidden_dim': 4096,
        'num_visual_tokens': 256,
        'decoder_shared_dim': 768,
        'num_phonemes': 70,
        'mel_bins': 80,
        
        # Training parameters
        'lr_stage1': args.lr_stage1,
        'lr_stage2': args.lr_stage2,
        'lr_stage3': args.lr_stage3,
        'weight_decay': args.weight_decay,
        
        # Loss weights
        'contrastive_weight': args.contrastive_weight,
        'text_weight': args.text_weight,
        'phoneme_weight': args.phoneme_weight,
        'audio_weight': args.audio_weight,
        'alignment_weight': args.alignment_weight,
    }
    
    # Save configuration
    config_path = Path(args.checkpoint_dir) / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved configuration to {config_path}")
    
    # Create trainer
    print("Initializing trainer...")
    trainer = NLinkTrainer(
        config=config,
        device=args.device,
        use_wandb=args.use_wandb,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    # Run training
    if args.stage == "all":
        # Run all stages sequentially
        print("\n=== Stage 1: MEG-Mel Alignment ===")
        trainer.train_stage1(train_loader, val_loader, num_epochs=args.epochs_stage1)
        
        print("\n=== Stage 2: Brain-to-Text Alignment ===")
        trainer.train_stage2(train_loader, val_loader, num_epochs=args.epochs_stage2)
        
        print("\n=== Stage 3: End-to-End Fine-tuning ===")
        trainer.train_stage3(train_loader, val_loader, num_epochs=args.epochs_stage3)
        
    elif args.stage == "stage1":
        print("\n=== Stage 1: MEG-Mel Alignment ===")
        trainer.train_stage1(train_loader, val_loader, num_epochs=args.epochs_stage1)
        
    elif args.stage == "stage2":
        print("\n=== Stage 2: Brain-to-Text Alignment ===")
        trainer.train_stage2(train_loader, val_loader, num_epochs=args.epochs_stage2)
        
    elif args.stage == "stage3":
        print("\n=== Stage 3: End-to-End Fine-tuning ===")
        trainer.train_stage3(train_loader, val_loader, num_epochs=args.epochs_stage3)
    
    print("\nTraining completed!")
    
    # Close W&B
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()