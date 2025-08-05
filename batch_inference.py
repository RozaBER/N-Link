#!/usr/bin/env python3
"""
N-Link Batch Inference Script
Process multiple MEG files or dataset samples in batch mode
"""

import argparse
import json
import time
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Optional
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from n_link.utils import RealTimeInference
from n_link.data import MASCMEGDataset


class BatchInferenceEngine:
    """Batch processing engine for N-Link inference"""
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: str = "cuda",
        batch_size: int = 8,
        num_workers: int = 4,
    ):
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Load configuration
        if config_path is None:
            config_path = Path(checkpoint_path).parent / "config.json"
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize inference engine
        self.inference_engine = RealTimeInference(
            checkpoint_path=checkpoint_path,
            config=self.config,
            device=device,
        )
        
        # Results storage
        self.results = []
    
    def process_dataset(
        self,
        dataset,
        indices: Optional[List[int]] = None,
        save_outputs: bool = True,
        output_dir: str = "./batch_outputs",
    ) -> pd.DataFrame:
        """Process multiple samples from a dataset"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine which samples to process
        if indices is None:
            indices = list(range(len(dataset)))
        
        print(f"Processing {len(indices)} samples...")
        
        # Process in batches
        results = []
        
        for batch_start in tqdm(range(0, len(indices), self.batch_size)):
            batch_indices = indices[batch_start:batch_start + self.batch_size]
            batch_results = self._process_batch(
                dataset, batch_indices, save_outputs, output_dir
            )
            results.extend(batch_results)
        
        # Create results DataFrame
        df = pd.DataFrame(results)
        
        # Save summary
        summary_path = output_dir / "batch_inference_summary.csv"
        df.to_csv(summary_path, index=False)
        print(f"\nSaved results summary to: {summary_path}")
        
        return df
    
    def _process_batch(
        self,
        dataset,
        indices: List[int],
        save_outputs: bool,
        output_dir: Path,
    ) -> List[Dict]:
        """Process a batch of samples"""
        batch_results = []
        
        # Load batch data
        batch_meg = []
        batch_metadata = []
        
        for idx in indices:
            sample = dataset[idx]
            meg_data = sample['meg_signal']
            
            # Store MEG data
            batch_meg.append(meg_data)
            
            # Store metadata
            metadata = {
                'index': idx,
                'subject_id': sample.get('subject_id', 'unknown'),
                'ground_truth_text': sample.get('text', ''),
            }
            batch_metadata.append(metadata)
        
        # Stack MEG data
        batch_meg_tensor = torch.stack([
            torch.from_numpy(meg.numpy() if hasattr(meg, 'numpy') else meg).float()
            for meg in batch_meg
        ]).to(self.device)
        
        # Run inference
        start_time = time.time()
        
        # Process each sample (could be optimized for true batch processing)
        for i, meg_tensor in enumerate(batch_meg_tensor):
            sample_start = time.time()
            
            # Add batch dimension
            meg_input = meg_tensor.unsqueeze(0)
            
            # Run inference
            outputs = self.inference_engine.streamer._process_meg_tensor(meg_input)
            
            # Calculate metrics
            inference_time = (time.time() - sample_start) * 1000  # ms
            
            # Prepare result
            result = {
                **batch_metadata[i],
                'inference_time_ms': inference_time,
                'predicted_text': outputs.get('text', ''),
                'num_phonemes': len(outputs.get('phonemes', [])),
                'audio_duration_s': outputs['audio'].shape[-1] / 16000 if 'audio' in outputs else 0,
            }
            
            # Calculate accuracy if ground truth available
            if result['ground_truth_text'] and result['predicted_text']:
                gt = result['ground_truth_text']
                pred = result['predicted_text']
                
                # Character-level accuracy
                correct_chars = sum(1 for a, b in zip(pred, gt) if a == b)
                result['char_accuracy'] = correct_chars / max(len(gt), 1)
                
                # Word-level accuracy
                gt_words = gt.split()
                pred_words = pred.split()
                correct_words = sum(1 for a, b in zip(pred_words, gt_words) if a == b)
                result['word_accuracy'] = correct_words / max(len(gt_words), 1)
            
            batch_results.append(result)
            
            # Save outputs if requested
            if save_outputs:
                sample_dir = output_dir / f"sample_{indices[i]:05d}"
                sample_dir.mkdir(exist_ok=True)
                
                # Save text
                text_path = sample_dir / "predicted_text.txt"
                with open(text_path, 'w') as f:
                    f.write(outputs.get('text', ''))
                
                # Save audio
                if 'audio' in outputs:
                    audio_path = sample_dir / "predicted_audio.wav"
                    audio_data = outputs['audio'].squeeze().cpu().numpy()
                    sf.write(audio_path, audio_data, 16000)
                
                # Save metadata
                meta_path = sample_dir / "metadata.json"
                with open(meta_path, 'w') as f:
                    json.dump(result, f, indent=2)
        
        return batch_results
    
    def process_files(
        self,
        file_paths: List[str],
        save_outputs: bool = True,
        output_dir: str = "./batch_outputs",
    ) -> pd.DataFrame:
        """Process multiple MEG files"""
        import mne
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        print(f"Processing {len(file_paths)} files...")
        
        for file_path in tqdm(file_paths):
            try:
                # Load MEG file
                raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
                picks = mne.pick_types(raw.info, meg=True, ref_meg=False)
                meg_data, _ = raw[picks, :]
                
                # Convert to tensor
                meg_tensor = torch.from_numpy(meg_data).float().unsqueeze(0).to(self.device)
                
                # Run inference
                start_time = time.time()
                outputs = self.inference_engine.streamer._process_meg_tensor(meg_tensor)
                inference_time = (time.time() - start_time) * 1000
                
                # Prepare result
                result = {
                    'file_path': file_path,
                    'file_name': Path(file_path).name,
                    'inference_time_ms': inference_time,
                    'predicted_text': outputs.get('text', ''),
                    'num_phonemes': len(outputs.get('phonemes', [])),
                    'audio_duration_s': outputs['audio'].shape[-1] / 16000 if 'audio' in outputs else 0,
                }
                
                results.append(result)
                
                # Save outputs
                if save_outputs:
                    file_output_dir = output_dir / Path(file_path).stem
                    file_output_dir.mkdir(exist_ok=True)
                    
                    # Save predictions
                    with open(file_output_dir / "predicted_text.txt", 'w') as f:
                        f.write(outputs.get('text', ''))
                    
                    if 'audio' in outputs:
                        audio_path = file_output_dir / "predicted_audio.wav"
                        audio_data = outputs['audio'].squeeze().cpu().numpy()
                        sf.write(audio_path, audio_data, 16000)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                results.append({
                    'file_path': file_path,
                    'file_name': Path(file_path).name,
                    'error': str(e),
                })
        
        # Create and save summary
        df = pd.DataFrame(results)
        summary_path = output_dir / "file_inference_summary.csv"
        df.to_csv(summary_path, index=False)
        print(f"\nSaved results summary to: {summary_path}")
        
        return df
    
    def benchmark_performance(
        self,
        dataset,
        num_samples: int = 100,
    ) -> Dict:
        """Benchmark inference performance"""
        print(f"Benchmarking on {num_samples} samples...")
        
        # Select random samples
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        
        # Measure performance
        inference_times = []
        meg_lengths = []
        
        for idx in tqdm(indices):
            sample = dataset[idx]
            meg_data = sample['meg_signal']
            meg_tensor = torch.from_numpy(
                meg_data.numpy() if hasattr(meg_data, 'numpy') else meg_data
            ).float().unsqueeze(0).to(self.device)
            
            # Time inference
            start_time = time.time()
            _ = self.inference_engine.streamer._process_meg_tensor(meg_tensor)
            inference_time = (time.time() - start_time) * 1000  # ms
            
            inference_times.append(inference_time)
            meg_lengths.append(meg_data.shape[1])  # Time dimension
        
        # Calculate statistics
        inference_times = np.array(inference_times)
        meg_lengths = np.array(meg_lengths)
        
        stats = {
            'num_samples': len(inference_times),
            'mean_inference_ms': np.mean(inference_times),
            'std_inference_ms': np.std(inference_times),
            'min_inference_ms': np.min(inference_times),
            'max_inference_ms': np.max(inference_times),
            'p50_inference_ms': np.percentile(inference_times, 50),
            'p95_inference_ms': np.percentile(inference_times, 95),
            'p99_inference_ms': np.percentile(inference_times, 99),
            'mean_meg_length': np.mean(meg_lengths),
            'real_time_factor': np.mean(meg_lengths / inference_times),
            'samples_per_second': 1000 / np.mean(inference_times),
        }
        
        return stats


def parse_args():
    parser = argparse.ArgumentParser(description="N-Link Batch Inference")
    
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
        "--mode",
        type=str,
        choices=["dataset", "files", "benchmark"],
        default="dataset",
        help="Inference mode",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/data/zshao/masc_meg",
        help="Path to MASC-MEG dataset (for dataset mode)",
    )
    parser.add_argument(
        "--file_list",
        type=str,
        help="Text file containing list of MEG files to process",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        help="Specific sample indices to process",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to process (default: all)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./batch_outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--save_outputs",
        action="store_true",
        help="Save predicted text and audio",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize batch inference engine
    print("Initializing batch inference engine...")
    engine = BatchInferenceEngine(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    if args.mode == "dataset":
        # Process dataset samples
        print("Loading dataset...")
        dataset = MASCMEGDataset(
            data_root=args.data_root,
            split='test',
            cache_dir='./cache',
        )
        
        # Determine which samples to process
        if args.indices:
            indices = args.indices
        elif args.num_samples:
            indices = list(range(min(args.num_samples, len(dataset))))
        else:
            indices = None  # Process all
        
        # Run batch inference
        results_df = engine.process_dataset(
            dataset=dataset,
            indices=indices,
            save_outputs=args.save_outputs,
            output_dir=args.output_dir,
        )
        
        # Print summary statistics
        print("\n" + "="*60)
        print("Batch Inference Summary")
        print("="*60)
        print(f"Total samples processed: {len(results_df)}")
        print(f"Average inference time: {results_df['inference_time_ms'].mean():.2f} ms")
        print(f"Min/Max inference time: {results_df['inference_time_ms'].min():.2f} / {results_df['inference_time_ms'].max():.2f} ms")
        
        if 'char_accuracy' in results_df.columns:
            print(f"Average character accuracy: {results_df['char_accuracy'].mean()*100:.1f}%")
            print(f"Average word accuracy: {results_df['word_accuracy'].mean()*100:.1f}%")
    
    elif args.mode == "files":
        # Process list of files
        if not args.file_list:
            print("Error: --file_list required for files mode")
            return
        
        # Read file list
        with open(args.file_list, 'r') as f:
            file_paths = [line.strip() for line in f if line.strip()]
        
        # Run batch inference
        results_df = engine.process_files(
            file_paths=file_paths,
            save_outputs=args.save_outputs,
            output_dir=args.output_dir,
        )
        
        # Print summary
        print("\n" + "="*60)
        print("File Processing Summary")
        print("="*60)
        print(f"Total files processed: {len(results_df)}")
        print(f"Successful: {len(results_df[~results_df.get('error', pd.Series()).notna()])}")
        print(f"Failed: {len(results_df[results_df.get('error', pd.Series()).notna()])}")
    
    elif args.mode == "benchmark":
        # Run performance benchmark
        print("Loading dataset for benchmarking...")
        dataset = MASCMEGDataset(
            data_root=args.data_root,
            split='test',
            cache_dir='./cache',
        )
        
        num_samples = args.num_samples or 100
        stats = engine.benchmark_performance(dataset, num_samples)
        
        # Print benchmark results
        print("\n" + "="*60)
        print("Performance Benchmark Results")
        print("="*60)
        for key, value in stats.items():
            print(f"{key}: {value:.2f}")
        
        # Save benchmark results
        benchmark_path = Path(args.output_dir) / "benchmark_results.json"
        benchmark_path.parent.mkdir(parents=True, exist_ok=True)
        with open(benchmark_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nSaved benchmark results to: {benchmark_path}")


if __name__ == "__main__":
    main()