#!/usr/bin/env python3
"""
N-Link Inference Script
Real-time MEG-to-Speech/Text inference and demo
"""

import argparse
import json
import time
from pathlib import Path
import torch
import numpy as np
import soundfile as sf
from typing import Optional, Dict
import matplotlib.pyplot as plt

from n_link.utils import RealTimeInference, MEGSimulator
from n_link.data import MASCMEGDataset


def parse_args():
    parser = argparse.ArgumentParser(description="N-Link Inference and Demo")
    
    # Model arguments
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
    
    # Input arguments
    parser.add_argument(
        "--mode",
        type=str,
        choices=["file", "stream", "simulate", "benchmark"],
        default="file",
        help="Inference mode",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="MEG file for file mode",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/data/zshao/masc_meg",
        help="MASC-MEG data root for loading test samples",
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory for saving outputs",
    )
    parser.add_argument(
        "--save_audio",
        action="store_true",
        help="Save generated audio",
    )
    parser.add_argument(
        "--save_text",
        action="store_true",
        help="Save generated text",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize MEG signals and outputs",
    )
    
    # Streaming arguments
    parser.add_argument(
        "--buffer_size_ms",
        type=int,
        default=250,
        help="Buffer size in milliseconds for streaming",
    )
    parser.add_argument(
        "--overlap_ms",
        type=int,
        default=50,
        help="Overlap in milliseconds for streaming",
    )
    
    # Other arguments
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


def load_config(args) -> Dict:
    """Load model configuration"""
    if args.config:
        config_path = Path(args.config)
    else:
        # Try to find config in checkpoint directory
        checkpoint_dir = Path(args.checkpoint).parent
        config_path = checkpoint_dir / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Add streaming parameters
    config['buffer_size_ms'] = args.buffer_size_ms
    config['overlap_ms'] = args.overlap_ms
    
    return config


def run_file_inference(args, inference_engine):
    """Run inference on a single MEG file"""
    print(f"Running inference on: {args.input_file}")
    
    # Load MEG data
    if args.input_file:
        # Load specific file
        import mne
        raw = mne.io.read_raw_fif(args.input_file, preload=True, verbose=False)
        picks = mne.pick_types(raw.info, meg=True, ref_meg=False)
        meg_data, times = raw[picks, :]
        sampling_rate = raw.info['sfreq']
    else:
        # Load a sample from test set
        print("Loading sample from test dataset...")
        dataset = MASCMEGDataset(
            data_root=args.data_root,
            split='test',
            preprocess=True,
        )
        sample = dataset[0]
        meg_data = sample['meg_signal'].numpy()
        sampling_rate = dataset.sampling_rate
    
    print(f"MEG data shape: {meg_data.shape}")
    print(f"Sampling rate: {sampling_rate} Hz")
    
    # Process entire signal
    start_time = time.time()
    
    # Convert to tensor
    meg_tensor = torch.from_numpy(meg_data).float().unsqueeze(0).to(args.device)
    
    # Run inference
    outputs = inference_engine.streamer._process_meg_tensor(meg_tensor)
    
    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.2f} seconds")
    
    # Display results
    print("\n=== Results ===")
    
    if 'text' in outputs:
        print(f"Generated text: {outputs['text']}")
    
    if 'phonemes' in outputs:
        print(f"Phonemes: {outputs['phonemes'][:20]}...")  # Show first 20
    
    if 'latency_ms' in outputs:
        print(f"Processing latency: {outputs['latency_ms']:.1f} ms")
    
    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.save_audio and 'audio' in outputs:
        audio_path = output_dir / "generated_audio.wav"
        audio_data = outputs['audio'].squeeze().cpu().numpy()
        sf.write(audio_path, audio_data, 16000)
        print(f"Saved audio to: {audio_path}")
    
    if args.save_text and 'text' in outputs:
        text_path = output_dir / "generated_text.txt"
        with open(text_path, 'w') as f:
            f.write(outputs['text'])
        print(f"Saved text to: {text_path}")
    
    if args.visualize:
        visualize_results(meg_data, outputs, output_dir)
    
    return outputs


def run_streaming_inference(args, inference_engine):
    """Run real-time streaming inference"""
    print("Starting streaming inference...")
    print("Press Ctrl+C to stop")
    
    # Start streaming processor
    inference_engine.start_streaming()
    
    # Create MEG simulator
    simulator = MEGSimulator(
        num_channels=208,
        sampling_rate=1000,
        chunk_duration_ms=50,
    )
    
    output_buffer = []
    
    try:
        while True:
            # Get simulated MEG chunk
            meg_chunk = simulator.get_chunk()
            
            # Process chunk
            outputs = inference_engine.process_meg_stream(meg_chunk)
            
            if outputs:
                output_buffer.append(outputs)
                
                # Display latest output
                if args.verbose:
                    print(f"\rLatency: {outputs.get('latency_ms', 0):.1f} ms", end='')
                
                # Save periodic outputs
                if len(output_buffer) % 20 == 0:  # Every second (assuming 50ms chunks)
                    if 'text' in outputs:
                        print(f"\nText: {outputs['text']}")
            
            # Simulate real-time delay
            time.sleep(0.05)  # 50ms
            
    except KeyboardInterrupt:
        print("\nStopping streaming...")
    
    # Stop streaming
    inference_engine.stop_streaming()
    
    # Save accumulated outputs
    if output_buffer and args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Concatenate audio if available
        if args.save_audio and 'audio' in output_buffer[0]:
            all_audio = []
            for out in output_buffer:
                if 'audio' in out:
                    all_audio.append(out['audio'].squeeze().cpu().numpy())
            
            if all_audio:
                combined_audio = np.concatenate(all_audio)
                audio_path = output_dir / "streaming_audio.wav"
                sf.write(audio_path, combined_audio, 16000)
                print(f"Saved streaming audio to: {audio_path}")


def run_simulation(args, inference_engine):
    """Run inference on simulated MEG data"""
    print("Running inference on simulated MEG data...")
    
    # Create simulator
    simulator = MEGSimulator(
        num_channels=208,
        sampling_rate=1000,
        chunk_duration_ms=args.buffer_size_ms,
    )
    
    # Generate longer signal
    num_chunks = 10
    all_chunks = []
    
    for i in range(num_chunks):
        chunk = simulator.get_chunk()
        all_chunks.append(chunk)
    
    # Concatenate chunks
    meg_data = np.concatenate(all_chunks, axis=1)
    print(f"Generated MEG data shape: {meg_data.shape}")
    
    # Run inference
    meg_tensor = torch.from_numpy(meg_data).float().unsqueeze(0).to(args.device)
    outputs = inference_engine.streamer._process_meg_tensor(meg_tensor)
    
    # Display results
    print("\n=== Simulation Results ===")
    if 'text' in outputs:
        print(f"Generated text: {outputs['text']}")
    if 'latency_ms' in outputs:
        print(f"Processing latency: {outputs['latency_ms']:.1f} ms")
    
    return outputs


def run_benchmark(args, inference_engine):
    """Benchmark inference performance"""
    print("Running performance benchmark...")
    
    # Benchmark latency
    stats = inference_engine.benchmark_latency(num_chunks=100)
    
    print("\n=== Benchmark Results ===")
    print(f"Mean latency: {stats['mean_ms']:.2f} ms")
    print(f"Std latency: {stats['std_ms']:.2f} ms")
    print(f"Min latency: {stats['min_ms']:.2f} ms")
    print(f"Max latency: {stats['max_ms']:.2f} ms")
    print(f"P50 latency: {stats['p50_ms']:.2f} ms")
    print(f"P95 latency: {stats['p95_ms']:.2f} ms")
    print(f"P99 latency: {stats['p99_ms']:.2f} ms")
    print(f"Real-time factor: {stats['real_time_factor']:.2f}x")
    print(f"Meets real-time: {'Yes' if stats['is_real_time'] else 'No'}")
    
    # Memory usage
    if torch.cuda.is_available():
        print(f"\nGPU memory used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    return stats


def visualize_results(meg_data: np.ndarray, outputs: Dict, output_dir: Path):
    """Visualize MEG signals and outputs"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot MEG signals (first 10 channels)
    ax = axes[0]
    time_points = np.arange(meg_data.shape[1]) / 1000  # Convert to seconds
    for i in range(min(10, meg_data.shape[0])):
        ax.plot(time_points, meg_data[i] + i*5, alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('MEG Channels')
    ax.set_title('MEG Signals (First 10 channels)')
    ax.grid(True, alpha=0.3)
    
    # Plot brain features if available
    if 'brain_features' in outputs:
        ax = axes[1]
        features = outputs['brain_features'].squeeze().cpu().numpy()
        if features.ndim == 2:
            im = ax.imshow(features.T, aspect='auto', origin='lower', cmap='viridis')
            ax.set_xlabel('Time')
            ax.set_ylabel('Feature Dimension')
            ax.set_title('Brain Encoder Features')
            plt.colorbar(im, ax=ax)
    
    # Plot audio waveform if available
    if 'audio' in outputs:
        ax = axes[2]
        audio = outputs['audio'].squeeze().cpu().numpy()
        audio_time = np.arange(len(audio)) / 16000  # 16kHz audio
        ax.plot(audio_time, audio)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Generated Audio Waveform')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = output_dir / "visualization.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {fig_path}")
    plt.close()


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize inference engine
    print("Loading model...")
    inference_engine = RealTimeInference(
        checkpoint_path=args.checkpoint,
        config=config,
        device=args.device,
    )
    
    # Run inference based on mode
    if args.mode == "file":
        run_file_inference(args, inference_engine)
    
    elif args.mode == "stream":
        run_streaming_inference(args, inference_engine)
    
    elif args.mode == "simulate":
        run_simulation(args, inference_engine)
    
    elif args.mode == "benchmark":
        run_benchmark(args, inference_engine)
    
    print("\nInference completed!")


if __name__ == "__main__":
    main()