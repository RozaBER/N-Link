#!/usr/bin/env python3
"""
N-Link Model Evaluation Script
Comprehensive evaluation of MEG-to-Speech/Text model performance
"""

import argparse
import json
import time
from pathlib import Path
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

# Metrics imports
from torchmetrics.text import BLEUScore, CharErrorRate, WordErrorRate
# Audio metrics - optional imports
try:
    from torchmetrics.audio import PerceptualEvaluationSpeechQuality, ShortTimeObjectiveIntelligibility
    AUDIO_METRICS_AVAILABLE = True
except ImportError:
    AUDIO_METRICS_AVAILABLE = False
import torch.nn.functional as F

from n_link.data import MASCMEGDataset, create_dataloaders
from n_link.training import NLinkTrainer
from n_link.utils import RealTimeInference


class NLinkEvaluator:
    """Comprehensive evaluator for N-Link model"""
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: str = "cuda",
        verbose: bool = True,
    ):
        self.device = device
        self.verbose = verbose
        
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Try to find config in checkpoint directory
            checkpoint_dir = Path(checkpoint_path).parent
            config_path = checkpoint_dir / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                raise ValueError(f"Config file not found. Please provide config_path.")
        
        # Initialize model
        self.model = self._load_model(checkpoint_path)
        
        # Initialize metrics
        self._init_metrics()
        
        # Results storage
        self.results = defaultdict(list)
    
    def _load_model(self, checkpoint_path: str):
        """Load trained model from checkpoint"""
        if self.verbose:
            print(f"Loading model from {checkpoint_path}")
        
        # Initialize trainer (which contains all model components)
        # Add num_subjects if not in config
        if 'num_subjects' not in self.config:
            self.config['num_subjects'] = 28  # MASC has subjects 1-27, need 28 for indexing
        
        trainer = NLinkTrainer(
            config=self.config,
            device=self.device,
            use_wandb=False,
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model states
        if "brain_encoder" in checkpoint:
            trainer.brain_encoder.load_state_dict(checkpoint["brain_encoder"])
        if "meg_llava_adapter" in checkpoint:
            trainer.meg_llava_adapter.load_state_dict(checkpoint["meg_llava_adapter"])
        if "multi_decoder" in checkpoint:
            trainer.multi_decoder.load_state_dict(checkpoint["multi_decoder"])
        if "meg_mel_aligner" in checkpoint:
            trainer.meg_mel_aligner.load_state_dict(checkpoint["meg_mel_aligner"])
        
        # Set to eval mode
        trainer.brain_encoder.eval()
        trainer.meg_llava_adapter.eval()
        trainer.multi_decoder.eval()
        trainer.meg_mel_aligner.eval()
        
        return trainer
    
    def _init_metrics(self):
        """Initialize evaluation metrics"""
        # Text metrics
        self.bleu = BLEUScore(n_gram=4)
        self.cer = CharErrorRate()
        self.wer = WordErrorRate()
        
        # Audio metrics (will be initialized when needed due to heavy dependencies)
        if AUDIO_METRICS_AVAILABLE:
            self.pesq = PerceptualEvaluationSpeechQuality(fs=16000, mode="wb")
            self.stoi = ShortTimeObjectiveIntelligibility(fs=16000, extended=False)
        else:
            self.pesq = None
            self.stoi = None
        
        # Storage for custom metrics
        self.custom_metrics = {}
    
    def evaluate_stage1(self, dataloader) -> Dict[str, float]:
        """Evaluate Stage 1: MEG-Mel alignment"""
        if self.verbose:
            print("\n=== Evaluating Stage 1: MEG-Mel Alignment ===")
        
        total_loss = 0
        total_similarity = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Stage 1 Evaluation"):
                meg_signal = batch["meg_signal"].to(self.device)
                audio = batch.get("audio")
                
                if audio is not None:
                    audio = audio.to(self.device)
                    
                    # Get MEG and audio features
                    outputs = self.model.meg_mel_aligner(meg_signal, audio=audio)
                    
                    # Contrastive loss
                    if "loss" in outputs:
                        total_loss += outputs["loss"].item()
                    
                    # Feature similarity
                    meg_features = outputs["meg_features"]
                    mel_features = outputs["mel_features"]
                    
                    # Normalize features
                    meg_features = F.normalize(meg_features, dim=-1)
                    mel_features = F.normalize(mel_features, dim=-1)
                    
                    # Compute cosine similarity
                    similarity = (meg_features * mel_features).sum(dim=-1).mean()
                    total_similarity += similarity.item()
                    
                    num_batches += 1
        
        results = {
            "stage1_loss": total_loss / max(num_batches, 1),
            "stage1_similarity": total_similarity / max(num_batches, 1),
        }
        
        if self.verbose:
            for k, v in results.items():
                print(f"  {k}: {v:.4f}")
        
        return results
    
    def evaluate_stage2(self, dataloader) -> Dict[str, float]:
        """Evaluate Stage 2: Brain-to-text alignment"""
        if self.verbose:
            print("\n=== Evaluating Stage 2: Brain-to-Text Alignment ===")
        
        total_loss = 0
        total_accuracy = 0
        all_brain_features = []
        all_text_features = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Stage 2 Evaluation"):
                meg_signal = batch["meg_signal"].to(self.device)
                subject_id = batch.get("subject_id")
                if subject_id is not None:
                    subject_id = subject_id.to(self.device)
                
                # Get brain features
                brain_features = self.model.brain_encoder(meg_signal, subject_id)
                
                # Get text features (from batch)
                if "text_features" in batch:
                    text_features = batch["text_features"].to(self.device)
                    
                    # Store for retrieval evaluation
                    all_brain_features.append(brain_features)
                    all_text_features.append(text_features)
                    
                    # Compute alignment metrics
                    brain_norm = F.normalize(brain_features, dim=-1)
                    text_norm = F.normalize(text_features, dim=-1)
                    
                    # Similarity matrix
                    similarity = torch.matmul(brain_norm, text_norm.T)
                    
                    # Accuracy (correct matches on diagonal)
                    labels = torch.arange(brain_features.shape[0], device=self.device)
                    predictions = similarity.argmax(dim=-1)
                    accuracy = (predictions == labels).float().mean()
                    total_accuracy += accuracy.item()
                    
                    num_batches += 1
        
        # Compute retrieval metrics if we have features
        retrieval_metrics = {}
        if all_brain_features and all_text_features:
            brain_features = torch.cat(all_brain_features, dim=0)
            text_features = torch.cat(all_text_features, dim=0)
            
            retrieval_metrics = self._compute_retrieval_metrics(
                brain_features, text_features
            )
        
        results = {
            "stage2_accuracy": total_accuracy / max(num_batches, 1),
            **retrieval_metrics,
        }
        
        if self.verbose:
            for k, v in results.items():
                print(f"  {k}: {v:.4f}")
        
        return results
    
    def evaluate_stage3(self, dataloader) -> Dict[str, float]:
        """Evaluate Stage 3: End-to-end performance"""
        if self.verbose:
            print("\n=== Evaluating Stage 3: End-to-End Performance ===")
        
        # Collect predictions and targets
        all_text_preds = []
        all_text_targets = []
        all_phoneme_preds = []
        all_phoneme_targets = []
        audio_metrics = []
        
        total_latency = 0
        num_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Stage 3 Evaluation"):
                batch_size = batch["meg_signal"].shape[0]
                
                # Prepare batch
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Time the forward pass
                start_time = time.time()
                
                # Get model outputs
                model_outputs = self._forward_stage3(batch)
                
                latency = (time.time() - start_time) * 1000 / batch_size  # ms per sample
                total_latency += latency * batch_size
                
                # Process text predictions
                if "text_logits" in model_outputs and "text" in batch:
                    text_preds = model_outputs["text_logits"].argmax(dim=-1)
                    # Convert to strings (simplified - in practice use proper tokenizer)
                    for i in range(batch_size):
                        pred_text = self._decode_text(text_preds[i])
                        target_text = batch["text"][i] if isinstance(batch["text"], list) else batch["text"]
                        all_text_preds.append(pred_text)
                        all_text_targets.append(target_text)
                
                # Process phoneme predictions
                if "phoneme_logits" in model_outputs and "phoneme_ids" in batch:
                    # Decode CTC outputs
                    phoneme_preds = model_outputs["phoneme_logits"].argmax(dim=-1)
                    all_phoneme_preds.extend(phoneme_preds.cpu().numpy())
                    all_phoneme_targets.extend(batch["phoneme_ids"].cpu().numpy())
                
                # Audio metrics (compute on subset to save time)
                if "audio" in model_outputs and "audio" in batch and len(audio_metrics) < 10:
                    pred_audio = model_outputs["audio"]
                    target_audio = batch["audio"]
                    
                    # Compute audio metrics
                    for i in range(min(batch_size, 2)):  # Only first 2 samples
                        metrics = self._compute_audio_metrics(
                            pred_audio[i], target_audio[i]
                        )
                        audio_metrics.append(metrics)
                
                num_samples += batch_size
        
        # Compute text metrics
        text_metrics = {}
        if all_text_preds and all_text_targets:
            text_metrics = {
                "bleu": self.bleu(all_text_preds, [[t] for t in all_text_targets]).item(),
                "cer": self.cer(all_text_preds, all_text_targets).item(),
                "wer": self.wer(all_text_preds, all_text_targets).item(),
            }
        
        # Compute phoneme metrics
        phoneme_metrics = {}
        if all_phoneme_preds and all_phoneme_targets:
            phoneme_metrics = self._compute_phoneme_metrics(
                all_phoneme_preds, all_phoneme_targets
            )
        
        # Average audio metrics
        avg_audio_metrics = {}
        if audio_metrics:
            for key in audio_metrics[0].keys():
                avg_audio_metrics[f"audio_{key}"] = np.mean([m[key] for m in audio_metrics])
        
        results = {
            "avg_latency_ms": total_latency / max(num_samples, 1),
            **text_metrics,
            **phoneme_metrics,
            **avg_audio_metrics,
        }
        
        if self.verbose:
            for k, v in results.items():
                print(f"  {k}: {v:.4f}")
        
        return results
    
    def _forward_stage3(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Forward pass through the full model"""
        meg_signal = batch["meg_signal"]
        subject_id = batch.get("subject_id")
        
        # Get brain features
        brain_features = self.model.brain_encoder(meg_signal, subject_id)
        
        # Through adapter
        adapter_outputs = self.model.meg_llava_adapter(brain_features)
        
        # Through LLaVA interface (simplified)
        llava_outputs = {
            "hidden_states": torch.randn(
                meg_signal.shape[0], 100, 
                self.config.get("llava_hidden_dim", 4096),
                device=self.device
            )
        }
        
        # Through multi-output decoder
        target_modalities = ["text", "phoneme", "audio"]
        decoder_outputs = self.model.multi_decoder(
            llava_outputs["hidden_states"],
            llava_outputs=llava_outputs,
            target_modalities=target_modalities,
        )
        
        return decoder_outputs
    
    def _decode_text(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text (simplified)"""
        # In practice, use proper tokenizer
        # For now, just convert to characters
        valid_ids = token_ids[token_ids >= 0]  # Remove padding
        text = ''.join([chr(int(tid) % 128) for tid in valid_ids])
        return text
    
    def _compute_retrieval_metrics(
        self, 
        query_features: torch.Tensor, 
        target_features: torch.Tensor
    ) -> Dict[str, float]:
        """Compute retrieval metrics"""
        # Normalize features
        query_features = F.normalize(query_features, dim=-1)
        target_features = F.normalize(target_features, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(query_features, target_features.T)
        
        # Get rankings
        _, indices = similarity.sort(dim=1, descending=True)
        
        # Find correct matches (assuming aligned pairs)
        correct_indices = torch.arange(len(query_features), device=self.device)
        ranks = (indices == correct_indices.unsqueeze(1)).nonzero()[:, 1] + 1
        
        # Compute metrics
        metrics = {
            "retrieval_r@1": (ranks == 1).float().mean().item(),
            "retrieval_r@5": (ranks <= 5).float().mean().item(),
            "retrieval_r@10": (ranks <= 10).float().mean().item(),
            "retrieval_median_rank": ranks.median().item(),
            "retrieval_mean_rank": ranks.float().mean().item(),
        }
        
        return metrics
    
    def _compute_phoneme_metrics(
        self, 
        predictions: List[np.ndarray], 
        targets: List[np.ndarray]
    ) -> Dict[str, float]:
        """Compute phoneme-level metrics"""
        # Simplified phoneme error rate
        total_errors = 0
        total_length = 0
        
        for pred, target in zip(predictions, targets):
            # Simple alignment (in practice use dynamic programming)
            errors = np.sum(pred[:len(target)] != target)
            total_errors += errors
            total_length += len(target)
        
        per = total_errors / max(total_length, 1)
        
        return {
            "phoneme_error_rate": per,
            "phoneme_accuracy": 1 - per,
        }
    
    def _compute_audio_metrics(
        self, 
        pred_audio: torch.Tensor, 
        target_audio: torch.Tensor
    ) -> Dict[str, float]:
        """Compute audio quality metrics"""
        # Convert to numpy
        pred_audio = pred_audio.squeeze().cpu().numpy()
        target_audio = target_audio.squeeze().cpu().numpy()
        
        # Ensure same length
        min_len = min(len(pred_audio), len(target_audio))
        pred_audio = pred_audio[:min_len]
        target_audio = target_audio[:min_len]
        
        # Basic metrics
        mse = np.mean((pred_audio - target_audio) ** 2)
        snr = 10 * np.log10(np.var(target_audio) / (mse + 1e-10))
        
        return {
            "mse": mse,
            "snr": snr,
        }
    
    def evaluate_all_stages(self, test_loader) -> Dict[str, float]:
        """Run full evaluation on all stages"""
        results = {}
        
        # Stage 1
        stage1_results = self.evaluate_stage1(test_loader)
        results.update(stage1_results)
        
        # Stage 2
        stage2_results = self.evaluate_stage2(test_loader)
        results.update(stage2_results)
        
        # Stage 3
        stage3_results = self.evaluate_stage3(test_loader)
        results.update(stage3_results)
        
        return results
    
    def save_results(self, results: Dict[str, float], output_path: str):
        """Save evaluation results"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_to_json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(v) for v in obj]
            return obj
        
        # Add metadata
        results_with_meta = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": self.config,
            "results": convert_to_json_serializable(results),
        }
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(results_with_meta, f, indent=2)
        
        if self.verbose:
            print(f"\nSaved results to {output_path}")
    
    def generate_report(self, results: Dict[str, float]) -> str:
        """Generate a formatted evaluation report"""
        report = []
        report.append("=" * 60)
        report.append("N-Link Model Evaluation Report")
        report.append("=" * 60)
        report.append("")
        
        # Group results by stage
        stage1_results = {k: v for k, v in results.items() if "stage1" in k}
        stage2_results = {k: v for k, v in results.items() if "stage2" in k or "retrieval" in k}
        stage3_results = {k: v for k, v in results.items() if k not in stage1_results and k not in stage2_results}
        
        # Stage 1 results
        if stage1_results:
            report.append("Stage 1: MEG-Mel Alignment")
            report.append("-" * 30)
            for k, v in stage1_results.items():
                report.append(f"  {k}: {v:.4f}")
            report.append("")
        
        # Stage 2 results
        if stage2_results:
            report.append("Stage 2: Brain-to-Text Alignment")
            report.append("-" * 30)
            for k, v in stage2_results.items():
                report.append(f"  {k}: {v:.4f}")
            report.append("")
        
        # Stage 3 results
        if stage3_results:
            report.append("Stage 3: End-to-End Performance")
            report.append("-" * 30)
            for k, v in stage3_results.items():
                report.append(f"  {k}: {v:.4f}")
            report.append("")
        
        # Performance summary
        report.append("Performance Summary")
        report.append("-" * 30)
        
        if "avg_latency_ms" in results:
            latency = results["avg_latency_ms"]
            report.append(f"  Average latency: {latency:.1f} ms")
            report.append(f"  Real-time capable: {'Yes' if latency < 100 else 'No'}")
        
        if "wer" in results:
            report.append(f"  Word Error Rate: {results['wer']*100:.1f}%")
        
        if "phoneme_accuracy" in results:
            report.append(f"  Phoneme Accuracy: {results['phoneme_accuracy']*100:.1f}%")
        
        report.append("=" * 60)
        
        return "\n".join(report)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate N-Link Model")
    
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
        help="Path to model config (if not in checkpoint dir)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/data/zshao/masc_meg",
        help="Path to MASC-MEG dataset",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--stages",
        type=str,
        nargs="+",
        choices=["stage1", "stage2", "stage3", "all"],
        default=["all"],
        help="Which stages to evaluate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluator
    print("Initializing evaluator...")
    evaluator = NLinkEvaluator(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        verbose=args.verbose,
    )
    
    # Create test dataloader
    print("Loading test dataset...")
    from n_link.data.masc_meg_dataset import MASCMEGDataset
    from torch.utils.data import DataLoader
    
    # For evaluation, we need the wrapper from stage3_train.py
    import sys
    sys.path.append('/scratch/masc/nlink')
    from stage3_train import Stage3DataWrapper
    
    test_dataset = MASCMEGDataset(
        data_root=args.data_root,
        split='test',
        cache_dir='./cache',
        meg_channels=evaluator.config.get('meg_channels', 208),
        sampling_rate=evaluator.config.get('sampling_rate', 1000),
    )
    
    # Wrap for Stage 3
    test_dataset = Stage3DataWrapper(test_dataset)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    print(f"Test set size: {len(test_dataset)} samples")
    
    # Run evaluation
    results = {}
    
    if "all" in args.stages:
        print("\nRunning full evaluation...")
        results = evaluator.evaluate_all_stages(test_loader)
    else:
        if "stage1" in args.stages:
            stage1_results = evaluator.evaluate_stage1(test_loader)
            results.update(stage1_results)
        
        if "stage2" in args.stages:
            stage2_results = evaluator.evaluate_stage2(test_loader)
            results.update(stage2_results)
        
        if "stage3" in args.stages:
            stage3_results = evaluator.evaluate_stage3(test_loader)
            results.update(stage3_results)
    
    # Generate and print report
    report = evaluator.generate_report(results)
    print("\n" + report)
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"evaluation_results_{timestamp}.json"
    evaluator.save_results(results, results_path)
    
    # Save report
    report_path = output_dir / f"evaluation_report_{timestamp}.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Saved report to {report_path}")
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()