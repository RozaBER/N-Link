#!/usr/bin/env python3
"""
Stage 3 Training Script: End-to-End Fine-tuning
This script performs end-to-end fine-tuning with multi-output decoding
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


class Stage3DataWrapper(Dataset):
    """Wrapper that adds required fields for Stage 3 training"""
    
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        batch = self.dataset[idx]
        
        # Stage 3 needs text_ids and proper tokenized data
        # For now, we'll create dummy IDs - in production you'd use a real tokenizer
        
        # Create dummy text_ids (would be from LLaVA tokenizer in production)
        text = batch.get('text', ['dummy text'])[0] if isinstance(batch.get('text', []), list) else batch.get('text', 'dummy text')
        # Simple character-level tokenization for testing
        # Make it 100 tokens to match the sequence length from multi-decoder
        text_ids = torch.tensor([ord(c) % 100 for c in text[:100]], dtype=torch.long)
        if len(text_ids) < 100:
            text_ids = torch.nn.functional.pad(text_ids, (0, 100 - len(text_ids)), value=-100)  # -100 is ignore_index
        batch['text_ids'] = text_ids
        
        # Add text features like Stage 2
        meg_signal = batch['meg_signal']  # (C, T)
        text_features = meg_signal.mean(dim=1)  # (C,) = (208,)
        if text_features.shape[0] < 384:
            text_features = torch.nn.functional.pad(text_features, (0, 384 - text_features.shape[0]))
        else:
            text_features = text_features[:384]
        batch['text_features'] = text_features
        
        # Ensure phoneme data is properly formatted
        if 'phoneme_ids' not in batch:
            batch['phoneme_ids'] = torch.randint(0, 70, (20,))  # Dummy phonemes
            
        # Fix phoneme lengths to match the actual output sequence length (100)
        batch['phoneme_lengths'] = torch.tensor(100)  # Match multi-decoder output length
        
        # Ensure phoneme target lengths are reasonable
        if 'phoneme_target_lengths' not in batch:
            batch['phoneme_target_lengths'] = torch.tensor(20)  # Length of phoneme_ids
        
        # Ensure we have all required fields
        if 'mel' not in batch:
            batch['mel'] = torch.randn(80, 26)  # Dummy mel spectrogram
            
        return batch


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3: End-to-End Fine-tuning")
    
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
        "--llava_hidden_dim",
        type=int,
        default=4096,
        help="LLaVA hidden dimension",
    )
    parser.add_argument(
        "--decoder_shared_dim",
        type=int,
        default=768,
        help="Shared dimension for multi-output decoder",
    )
    parser.add_argument(
        "--num_phonemes",
        type=int,
        default=70,
        help="Number of phoneme classes",
    )
    parser.add_argument(
        "--audio_chunk_size",
        type=int,
        default=50,
        help="Audio synthesis chunk size (ms)",
    )
    
    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size (smaller for Stage 3 due to multi-output)",
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
        default=1e-5,
        help="Initial learning rate",
    )
    
    # Loss weights
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
    
    # Stage 2 checkpoint
    parser.add_argument(
        "--stage2_checkpoint",
        type=str,
        required=True,
        help="Path to Stage 2 checkpoint directory",
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
        default="n-link-stage3",
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
        "--gradient_checkpointing",
        action="store_true",
        help="Use gradient checkpointing to save memory",
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
        args.experiment_name = f"stage3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
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
    
    print(f"Stage 3 Training Configuration:")
    print(f"  Experiment: {args.experiment_name}")
    print(f"  Data root: {args.data_root}")
    print(f"  Stage 2 checkpoint: {args.stage2_checkpoint}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Device: {args.device}")
    print(f"  Loss weights: text={args.text_weight}, phoneme={args.phoneme_weight}, audio={args.audio_weight}, align={args.alignment_weight}")
    print(f"  Gradient checkpointing: {args.gradient_checkpointing}")
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
        
        # Wrap datasets for Stage 3
        train_dataset = Stage3DataWrapper(train_dataset)
        val_dataset = Stage3DataWrapper(val_dataset)
        test_dataset = Stage3DataWrapper(test_dataset)
        
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
        "llava_hidden_dim": args.llava_hidden_dim,
        "decoder_shared_dim": args.decoder_shared_dim,
        "num_phonemes": args.num_phonemes,
        "audio_chunk_size": args.audio_chunk_size,
        
        # Training parameters
        "learning_rate_stage3": args.learning_rate,
        "text_weight": args.text_weight,
        "phoneme_weight": args.phoneme_weight,
        "audio_weight": args.audio_weight,
        "alignment_weight": args.alignment_weight,
        "gradient_checkpointing": args.gradient_checkpointing,
        
        # Other parameters
        "device": args.device,
        "stage2_checkpoint": args.stage2_checkpoint,
    }
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = NLinkTrainer(
        config=trainer_config,
        device=args.device,
        use_wandb=args.use_wandb,
        checkpoint_dir=str(checkpoint_dir),
    )
    
    # Load Stage 2 checkpoint
    print(f"\nLoading Stage 2 checkpoint from {args.stage2_checkpoint}...")
    try:
        # Load checkpoint manually since _load_checkpoint doesn't accept checkpoint_path
        checkpoint = torch.load(args.stage2_checkpoint, map_location=args.device)
        
        # Load all available model weights
        if "meg_mel_aligner" in checkpoint:
            trainer.meg_mel_aligner.load_state_dict(checkpoint["meg_mel_aligner"])
            print("Loaded MEG-Mel aligner weights")
            
        if "brain_encoder" in checkpoint:
            trainer.brain_encoder.load_state_dict(checkpoint["brain_encoder"])
            print("Loaded brain encoder weights")
            
        if "meg_llava_adapter" in checkpoint:
            trainer.meg_llava_adapter.load_state_dict(checkpoint["meg_llava_adapter"])
            print("Loaded MEG-LLaVA adapter weights")
            
        if "multi_decoder" in checkpoint:
            trainer.multi_decoder.load_state_dict(checkpoint["multi_decoder"])
            print("Loaded multi-output decoder weights")
            
        print("Stage 2 checkpoint loaded successfully")
    except Exception as e:
        print(f"Error loading Stage 2 checkpoint: {e}")
        raise
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        print("\nGradient checkpointing requested but not implemented in MEGLLaVAInterface")
        # trainer.llava_interface.enable_gradient_checkpointing()
    
    # Train Stage 3
    print("\nStarting Stage 3 training...")
    try:
        trainer.train_stage3(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
        )
        print("\nStage 3 training completed successfully!")
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