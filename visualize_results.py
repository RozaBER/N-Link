#!/usr/bin/env python3
"""
N-Link Visualization Tools
Visualize model predictions, attention maps, and embeddings
"""

import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from typing import Dict, List, Optional, Tuple
import mne
from matplotlib.gridspec import GridSpec
import librosa
import librosa.display

from n_link.data import MASCMEGDataset
from n_link.training import NLinkTrainer
from n_link.utils import RealTimeInference


class NLinkVisualizer:
    """Visualization tools for N-Link model analysis"""
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: str = "cuda",
        output_dir: str = "./visualizations",
    ):
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.inference_engine = RealTimeInference(
            checkpoint_path=checkpoint_path,
            config=self._load_config(config_path, checkpoint_path),
            device=device,
        )
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def _load_config(self, config_path: Optional[str], checkpoint_path: str) -> Dict:
        """Load configuration"""
        import json
        
        if config_path:
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Try to find config in checkpoint directory
            checkpoint_dir = Path(checkpoint_path).parent
            config_file = checkpoint_dir / "config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    return json.load(f)
            else:
                raise ValueError("Config file not found")
    
    def visualize_meg_predictions(
        self,
        meg_data: np.ndarray,
        predictions: Dict,
        sample_name: str = "sample",
        save: bool = True,
    ):
        """Visualize MEG signals with model predictions"""
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(4, 2, figure=fig, height_ratios=[2, 1, 1, 1])
        
        # 1. MEG signals (butterfly plot)
        ax1 = fig.add_subplot(gs[0, :])
        time_points = np.arange(meg_data.shape[1]) / 1000  # Convert to seconds
        
        # Plot all channels with transparency
        for i in range(meg_data.shape[0]):
            ax1.plot(time_points, meg_data[i], alpha=0.3, linewidth=0.5)
        
        # Highlight a few channels
        for i in range(min(5, meg_data.shape[0])):
            ax1.plot(time_points, meg_data[i], linewidth=2, label=f'Ch {i+1}')
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('MEG Signal (fT)')
        ax1.set_title('MEG Signals (Butterfly Plot)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 2. Brain features heatmap
        if 'brain_features' in predictions:
            ax2 = fig.add_subplot(gs[1, 0])
            features = predictions['brain_features'].squeeze().cpu().numpy()
            if features.ndim == 2:
                im = ax2.imshow(features.T, aspect='auto', origin='lower', 
                               cmap='RdBu_r', interpolation='nearest')
                ax2.set_xlabel('Time Steps')
                ax2.set_ylabel('Feature Dimension')
                ax2.set_title('Brain Encoder Features')
                plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        
        # 3. Attention weights (if available)
        if 'attention_weights' in predictions:
            ax3 = fig.add_subplot(gs[1, 1])
            attention = predictions['attention_weights'].squeeze().cpu().numpy()
            if attention.ndim == 2:
                im = ax3.imshow(attention, aspect='auto', origin='lower',
                               cmap='Blues', interpolation='nearest')
                ax3.set_xlabel('MEG Channels')
                ax3.set_ylabel('Time Steps')
                ax3.set_title('Spatial Attention Weights')
                plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        
        # 4. Predicted text
        if 'text' in predictions:
            ax4 = fig.add_subplot(gs[2, :])
            ax4.text(0.5, 0.5, f"Predicted Text: {predictions['text']}", 
                    ha='center', va='center', fontsize=14, wrap=True,
                    transform=ax4.transAxes)
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            ax4.set_title('Generated Text')
        
        # 5. Audio waveform comparison
        if 'audio' in predictions:
            ax5 = fig.add_subplot(gs[3, :])
            audio = predictions['audio'].squeeze().cpu().numpy()
            audio_time = np.arange(len(audio)) / 16000  # 16kHz
            ax5.plot(audio_time, audio, linewidth=1, alpha=0.8)
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Amplitude')
            ax5.set_title('Generated Audio Waveform')
            ax5.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / f"{sample_name}_predictions.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        return fig
    
    def visualize_embeddings(
        self,
        embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        method: str = "tsne",
        title: str = "Brain Embeddings",
        save_name: str = "embeddings",
    ):
        """Visualize high-dimensional embeddings using t-SNE or PCA"""
        if embeddings.shape[0] > 5000:
            # Subsample for efficiency
            indices = np.random.choice(embeddings.shape[0], 5000, replace=False)
            embeddings = embeddings[indices]
            if labels is not None:
                labels = labels[indices]
        
        # Reduce dimensionality
        if method == "tsne":
            reducer = TSNE(n_components=2, perplexity=30, random_state=42)
            embedded = reducer.fit_transform(embeddings)
        elif method == "pca":
            reducer = PCA(n_components=2)
            embedded = reducer.fit_transform(embeddings)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        if labels is not None:
            # Color by labels
            scatter = plt.scatter(embedded[:, 0], embedded[:, 1], 
                                c=labels, cmap='tab20', alpha=0.6, s=20)
            plt.colorbar(scatter, label='Subject ID')
        else:
            plt.scatter(embedded[:, 0], embedded[:, 1], alpha=0.6, s=20)
        
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        save_path = self.output_dir / f"{save_name}_{method}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved embeddings visualization to {save_path}")
        
        return embedded
    
    def visualize_phoneme_confusion(
        self,
        predictions: List[np.ndarray],
        targets: List[np.ndarray],
        phoneme_vocab: Optional[List[str]] = None,
        save_name: str = "phoneme_confusion",
    ):
        """Create phoneme confusion matrix"""
        # Flatten predictions and targets
        all_preds = []
        all_targets = []
        
        for pred, target in zip(predictions, targets):
            min_len = min(len(pred), len(target))
            all_preds.extend(pred[:min_len])
            all_targets.extend(target[:min_len])
        
        # Get unique phonemes
        unique_phonemes = sorted(set(all_preds + all_targets))
        n_phonemes = len(unique_phonemes)
        
        # Create confusion matrix
        confusion = np.zeros((n_phonemes, n_phonemes))
        for pred, target in zip(all_preds, all_targets):
            pred_idx = unique_phonemes.index(pred)
            target_idx = unique_phonemes.index(target)
            confusion[target_idx, pred_idx] += 1
        
        # Normalize by row (true labels)
        confusion = confusion / (confusion.sum(axis=1, keepdims=True) + 1e-10)
        
        # Plot
        plt.figure(figsize=(12, 10))
        
        # Use phoneme labels if provided
        if phoneme_vocab and len(phoneme_vocab) >= n_phonemes:
            labels = [phoneme_vocab[i] for i in unique_phonemes]
        else:
            labels = [str(i) for i in unique_phonemes]
        
        sns.heatmap(confusion, annot=False, cmap='Blues', 
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Probability'})
        
        plt.xlabel('Predicted Phoneme')
        plt.ylabel('True Phoneme')
        plt.title('Phoneme Confusion Matrix')
        plt.tight_layout()
        
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved phoneme confusion matrix to {save_path}")
    
    def visualize_audio_comparison(
        self,
        predicted_audio: np.ndarray,
        target_audio: np.ndarray,
        sample_rate: int = 16000,
        save_name: str = "audio_comparison",
    ):
        """Compare predicted and target audio with spectrograms"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Ensure same length
        min_len = min(len(predicted_audio), len(target_audio))
        predicted_audio = predicted_audio[:min_len]
        target_audio = target_audio[:min_len]
        
        # Time axis
        time = np.arange(min_len) / sample_rate
        
        # 1. Waveforms
        ax = axes[0, 0]
        ax.plot(time, target_audio, alpha=0.7, label='Target', linewidth=1)
        ax.plot(time, predicted_audio, alpha=0.7, label='Predicted', linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Audio Waveforms')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Difference
        ax = axes[0, 1]
        difference = target_audio - predicted_audio
        ax.plot(time, difference, color='red', alpha=0.7, linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude Difference')
        ax.set_title('Waveform Difference (Target - Predicted)')
        ax.grid(True, alpha=0.3)
        
        # 3. Target spectrogram
        ax = axes[1, 0]
        D_target = librosa.stft(target_audio, n_fft=1024, hop_length=256)
        D_target_db = librosa.amplitude_to_db(np.abs(D_target), ref=np.max)
        img = librosa.display.specshow(D_target_db, sr=sample_rate, 
                                      x_axis='time', y_axis='hz', ax=ax)
        ax.set_title('Target Spectrogram')
        plt.colorbar(img, ax=ax, format='%+2.0f dB')
        
        # 4. Predicted spectrogram
        ax = axes[1, 1]
        D_pred = librosa.stft(predicted_audio, n_fft=1024, hop_length=256)
        D_pred_db = librosa.amplitude_to_db(np.abs(D_pred), ref=np.max)
        img = librosa.display.specshow(D_pred_db, sr=sample_rate,
                                      x_axis='time', y_axis='hz', ax=ax)
        ax.set_title('Predicted Spectrogram')
        plt.colorbar(img, ax=ax, format='%+2.0f dB')
        
        plt.tight_layout()
        
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved audio comparison to {save_path}")
    
    def visualize_training_curves(
        self,
        log_file: str,
        metrics: List[str] = ["loss", "accuracy"],
        save_name: str = "training_curves",
    ):
        """Plot training curves from log file"""
        # Read log data (assuming CSV or similar format)
        try:
            df = pd.read_csv(log_file)
        except:
            print(f"Could not read log file: {log_file}")
            return
        
        # Create subplots
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4*n_metrics), squeeze=False)
        
        for i, metric in enumerate(metrics):
            ax = axes[i, 0]
            
            # Find columns containing the metric
            train_cols = [col for col in df.columns if metric in col and 'train' in col]
            val_cols = [col for col in df.columns if metric in col and 'val' in col]
            
            # Plot training curves
            for col in train_cols:
                ax.plot(df.index, df[col], label=f'Train {col}', alpha=0.8)
            
            # Plot validation curves
            for col in val_cols:
                ax.plot(df.index, df[col], label=f'Val {col}', alpha=0.8, linestyle='--')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    def create_summary_figure(
        self,
        sample_data: Dict,
        save_name: str = "summary",
    ):
        """Create a comprehensive summary figure"""
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 3, figure=fig)
        
        # Use sample data to create various visualizations
        # This is a template - adapt based on your specific needs
        
        # 1. MEG signal overview
        ax1 = fig.add_subplot(gs[0, :])
        ax1.set_title("MEG Signal Overview", fontsize=16)
        # Add MEG visualization here
        
        # 2. Model architecture diagram (placeholder)
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.text(0.5, 0.5, "Model\nArchitecture", ha='center', va='center', 
                fontsize=14, transform=ax2.transAxes)
        ax2.set_title("N-Link Architecture")
        ax2.axis('off')
        
        # 3. Performance metrics
        ax3 = fig.add_subplot(gs[1, 1])
        if 'metrics' in sample_data:
            metrics = sample_data['metrics']
            labels = list(metrics.keys())
            values = list(metrics.values())
            ax3.bar(labels, values)
            ax3.set_title("Performance Metrics")
            ax3.set_ylabel("Score")
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # 4. Sample predictions
        ax4 = fig.add_subplot(gs[1, 2])
        if 'predictions' in sample_data:
            ax4.text(0.1, 0.5, f"Sample Predictions:\n{sample_data['predictions']}", 
                    fontsize=12, transform=ax4.transAxes, wrap=True)
        ax4.set_title("Sample Output")
        ax4.axis('off')
        
        # Add more visualizations as needed...
        
        plt.suptitle("N-Link Model Summary", fontsize=20)
        plt.tight_layout()
        
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved summary figure to {save_path}")
        
        return fig


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize N-Link Model Results")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model config",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/data/zshao/masc_meg",
        help="Path to MASC-MEG dataset",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Sample index to visualize",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./visualizations",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize visualizer
    print("Initializing visualizer...")
    visualizer = NLinkVisualizer(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        output_dir=args.output_dir,
    )
    
    # Load a sample from the dataset
    print("Loading sample data...")
    dataset = MASCMEGDataset(
        data_root=args.data_root,
        split='test',
        cache_dir='./cache',
    )
    
    sample = dataset[args.sample_idx]
    meg_data = sample['meg_signal'].numpy()
    
    # Run inference
    print("Running inference...")
    meg_tensor = torch.from_numpy(meg_data).float().unsqueeze(0).to(args.device)
    predictions = visualizer.inference_engine.streamer._process_meg_tensor(meg_tensor)
    
    # Create visualizations
    print("Creating visualizations...")
    
    # 1. MEG predictions visualization
    visualizer.visualize_meg_predictions(
        meg_data=meg_data,
        predictions=predictions,
        sample_name=f"sample_{args.sample_idx}",
    )
    
    # 2. If we have embeddings, visualize them
    if 'brain_features' in predictions:
        embeddings = predictions['brain_features'].cpu().numpy()
        visualizer.visualize_embeddings(
            embeddings=embeddings.reshape(-1, embeddings.shape[-1]),
            method="tsne",
            title="Brain Feature Embeddings",
            save_name=f"embeddings_sample_{args.sample_idx}",
        )
    
    # 3. Audio comparison if available
    if 'audio' in predictions and 'audio' in sample:
        visualizer.visualize_audio_comparison(
            predicted_audio=predictions['audio'].squeeze().cpu().numpy(),
            target_audio=sample['audio'].numpy(),
            save_name=f"audio_comparison_sample_{args.sample_idx}",
        )
    
    print(f"\nVisualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()