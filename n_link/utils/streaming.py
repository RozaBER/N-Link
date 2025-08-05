import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, List
from collections import deque
import threading
import queue
import time


class StreamingMEGProcessor:
    """
    Real-time MEG signal processor with streaming capabilities
    Handles buffering, overlap-add, and efficient inference
    """
    
    def __init__(
        self,
        models: Dict[str, nn.Module],
        config: Dict,
        device: str = "cuda",
    ):
        self.models = models
        self.config = config
        self.device = device
        
        # Buffer parameters
        self.buffer_size_ms = config.get("buffer_size_ms", 250)
        self.overlap_ms = config.get("overlap_ms", 50)
        self.sampling_rate = config.get("sampling_rate", 1000)
        
        # Calculate buffer sizes in samples
        self.buffer_size = int(self.buffer_size_ms * self.sampling_rate / 1000)
        self.overlap_size = int(self.overlap_ms * self.sampling_rate / 1000)
        self.hop_size = self.buffer_size - self.overlap_size
        
        # MEG parameters
        self.meg_channels = config.get("meg_channels", 208)
        
        # Initialize buffers
        self.meg_buffer = deque(maxlen=self.buffer_size)
        self.feature_buffer = deque(maxlen=10)  # Store recent features
        
        # KV cache for efficient LLM inference
        self.kv_cache = {}
        
        # Audio synthesis buffer
        self.audio_buffer = torch.zeros(1, 1, 0).to(device)
        
        # Threading for real-time processing
        self.processing_queue = queue.Queue(maxsize=5)
        self.output_queue = queue.Queue(maxsize=10)
        self.is_running = False
        
        # Set models to eval mode
        for model in self.models.values():
            model.eval()
    
    def start(self):
        """Start streaming processing threads"""
        self.is_running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()
        
        print("Streaming processor started")
    
    def stop(self):
        """Stop streaming processing"""
        self.is_running = False
        
        # Wait for threads to finish
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join()
        
        print("Streaming processor stopped")
    
    def process_chunk(self, meg_chunk: np.ndarray) -> Optional[Dict]:
        """
        Process a chunk of MEG data
        
        Args:
            meg_chunk: (channels, samples) MEG data chunk
            
        Returns:
            Dictionary with processed outputs or None if buffer not ready
        """
        # Add to buffer
        self.meg_buffer.extend(meg_chunk.T)  # Transpose to (samples, channels)
        
        # Check if we have enough data
        if len(self.meg_buffer) < self.buffer_size:
            return None
        
        # Extract buffer for processing
        buffer_array = np.array(self.meg_buffer)
        meg_tensor = torch.from_numpy(buffer_array).float().to(self.device)
        meg_tensor = meg_tensor.T.unsqueeze(0)  # (1, channels, samples)
        
        # Add to processing queue
        try:
            self.processing_queue.put_nowait(meg_tensor)
        except queue.Full:
            print("Processing queue full, dropping frame")
        
        # Try to get output
        try:
            output = self.output_queue.get_nowait()
            return output
        except queue.Empty:
            return None
    
    def _processing_loop(self):
        """Background processing loop"""
        while self.is_running:
            try:
                # Get MEG data from queue
                meg_tensor = self.processing_queue.get(timeout=0.1)
                
                # Process through pipeline
                output = self._process_meg_tensor(meg_tensor)
                
                # Add to output queue
                try:
                    self.output_queue.put_nowait(output)
                except queue.Full:
                    # Drop oldest output
                    try:
                        self.output_queue.get_nowait()
                        self.output_queue.put_nowait(output)
                    except queue.Empty:
                        pass
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
    
    @torch.no_grad()
    def _process_meg_tensor(self, meg_tensor: torch.Tensor) -> Dict:
        """
        Process MEG tensor through the model pipeline
        
        Args:
            meg_tensor: (1, channels, samples) MEG data
            
        Returns:
            Dictionary with all outputs
        """
        start_time = time.time()
        outputs = {}
        
        # 1. Extract features with brain encoder
        brain_features = self.models['brain_encoder'](
            meg_tensor,
            return_sequence=True
        )
        outputs['brain_features'] = brain_features
        
        # 2. Convert to LLaVA visual tokens
        adapter_output = self.models['meg_llava_adapter'](brain_features)
        visual_tokens = adapter_output['visual_tokens']
        
        # 3. Process through LLaVA with KV cache (simplified)
        # In real implementation, this would use actual LLaVA inference
        llava_hidden = self._cached_llava_forward(visual_tokens)
        
        # 4. Generate outputs
        decoder_outputs = self.models['multi_decoder'](
            llava_hidden,
            target_modalities=['text', 'phoneme', 'audio'],
            streaming=True
        )
        outputs.update(decoder_outputs)
        
        # 5. Post-process outputs
        if 'text_logits' in decoder_outputs:
            outputs['text'] = self._decode_text(decoder_outputs['text_logits'])
        
        if 'phoneme_logits' in decoder_outputs:
            outputs['phonemes'] = self._decode_phonemes(decoder_outputs['phoneme_logits'])
        
        # Calculate latency
        outputs['latency_ms'] = (time.time() - start_time) * 1000
        
        return outputs
    
    def _cached_llava_forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """
        Simplified cached forward pass for LLaVA
        In real implementation, this would use actual KV caching
        """
        # For now, return dummy hidden states
        B, T, D = visual_tokens.shape
        hidden_dim = self.config.get('llava_hidden_dim', 4096)
        
        # Simple projection as placeholder
        if not hasattr(self, '_llava_proj'):
            self._llava_proj = nn.Linear(D, hidden_dim).to(self.device)
        
        return self._llava_proj(visual_tokens)
    
    def _decode_text(self, logits: torch.Tensor) -> str:
        """Decode text from logits"""
        # Simple greedy decoding
        tokens = logits.argmax(dim=-1)
        
        # In real implementation, use tokenizer
        # For now, return placeholder
        return f"[Decoded text from {tokens.shape[1]} tokens]"
    
    def _decode_phonemes(self, logits: torch.Tensor) -> List[int]:
        """Decode phonemes using CTC"""
        # Get most likely phonemes
        predictions = logits.argmax(dim=-1).squeeze(0)
        
        # Remove blanks and repetitions
        phonemes = []
        prev = -1
        for p in predictions:
            if p != 0 and p != prev:  # 0 is blank token
                phonemes.append(p.item())
            prev = p
        
        return phonemes
    
    def reset_buffers(self):
        """Reset all buffers for new session"""
        self.meg_buffer.clear()
        self.feature_buffer.clear()
        self.kv_cache.clear()
        self.audio_buffer = torch.zeros(1, 1, 0).to(self.device)
        
        # Reset audio synthesizer buffer
        if 'multi_decoder' in self.models:
            self.models['multi_decoder'].audio_synthesizer.reset_streaming_buffer()


class RealTimeInference:
    """
    Complete real-time inference pipeline
    Manages model loading and streaming processing
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config: Dict,
        device: str = "cuda",
    ):
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.device = device
        
        # Load models
        self._load_models()
        
        # Initialize streaming processor
        self.streamer = StreamingMEGProcessor(
            models=self.models,
            config=config,
            device=device,
        )
    
    def _load_models(self):
        """Load models from checkpoint"""
        from ..models import (
            MEGBrainEncoder,
            MEGToLLaVAAdapter,
            MultiOutputDecoder,
        )
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Initialize models
        self.models = {}
        
        # Brain encoder
        self.models['brain_encoder'] = MEGBrainEncoder(
            meg_channels=self.config.get('meg_channels', 208),
            sampling_rate=self.config.get('sampling_rate', 1000),
            output_dim=self.config.get('brain_encoder_dim', 384),
        ).to(self.device)
        
        if 'brain_encoder' in checkpoint:
            self.models['brain_encoder'].load_state_dict(checkpoint['brain_encoder'])
        
        # MEG-LLaVA adapter
        self.models['meg_llava_adapter'] = MEGToLLaVAAdapter(
            meg_feature_dim=self.config.get('brain_encoder_dim', 384),
            llava_visual_dim=self.config.get('llava_visual_dim', 576),
        ).to(self.device)
        
        if 'meg_llava_adapter' in checkpoint:
            self.models['meg_llava_adapter'].load_state_dict(checkpoint['meg_llava_adapter'])
        
        # Multi-output decoder
        self.models['multi_decoder'] = MultiOutputDecoder(
            llava_hidden_dim=self.config.get('llava_hidden_dim', 4096),
            shared_dim=self.config.get('decoder_shared_dim', 768),
        ).to(self.device)
        
        if 'multi_decoder' in checkpoint:
            self.models['multi_decoder'].load_state_dict(checkpoint['multi_decoder'])
        
        # Set to eval mode
        for model in self.models.values():
            model.eval()
        
        print(f"Loaded models from {self.checkpoint_path}")
    
    def start_streaming(self):
        """Start real-time streaming"""
        self.streamer.start()
    
    def stop_streaming(self):
        """Stop streaming"""
        self.streamer.stop()
    
    def process_meg_stream(self, meg_chunk: np.ndarray) -> Optional[Dict]:
        """
        Process MEG data stream
        
        Args:
            meg_chunk: (channels, samples) MEG data
            
        Returns:
            Processed outputs or None
        """
        return self.streamer.process_chunk(meg_chunk)
    
    def benchmark_latency(self, num_chunks: int = 100) -> Dict[str, float]:
        """
        Benchmark processing latency
        
        Args:
            num_chunks: Number of chunks to process
            
        Returns:
            Latency statistics
        """
        latencies = []
        
        # Generate dummy MEG data
        chunk_size = int(self.config['buffer_size_ms'] * 
                        self.config['sampling_rate'] / 1000)
        
        for _ in range(num_chunks):
            # Create dummy MEG chunk
            meg_chunk = np.random.randn(self.config['meg_channels'], chunk_size)
            
            # Process
            start_time = time.time()
            output = self.streamer._process_meg_tensor(
                torch.from_numpy(meg_chunk).float().unsqueeze(0).to(self.device)
            )
            latency = (time.time() - start_time) * 1000
            
            latencies.append(latency)
        
        # Calculate statistics
        latencies = np.array(latencies)
        stats = {
            'mean_ms': np.mean(latencies),
            'std_ms': np.std(latencies),
            'min_ms': np.min(latencies),
            'max_ms': np.max(latencies),
            'p50_ms': np.percentile(latencies, 50),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99),
        }
        
        # Check if meets real-time requirement
        buffer_duration_ms = self.config['buffer_size_ms']
        stats['real_time_factor'] = buffer_duration_ms / stats['mean_ms']
        stats['is_real_time'] = stats['real_time_factor'] > 1.0
        
        return stats


class MEGSimulator:
    """Simulate MEG data stream for testing"""
    
    def __init__(
        self,
        num_channels: int = 208,
        sampling_rate: int = 1000,
        chunk_duration_ms: int = 50,
    ):
        self.num_channels = num_channels
        self.sampling_rate = sampling_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.chunk_size = int(chunk_duration_ms * sampling_rate / 1000)
        
        # Simulation parameters
        self.time = 0
        self.frequencies = np.random.uniform(1, 40, num_channels)  # 1-40 Hz
        self.phases = np.random.uniform(0, 2*np.pi, num_channels)
        self.amplitudes = np.random.uniform(0.5, 2.0, num_channels)
    
    def get_chunk(self) -> np.ndarray:
        """Generate next chunk of simulated MEG data"""
        # Time vector for this chunk
        t = np.linspace(
            self.time,
            self.time + self.chunk_duration_ms / 1000,
            self.chunk_size
        )
        
        # Generate sinusoidal signals
        chunk = np.zeros((self.num_channels, self.chunk_size))
        for i in range(self.num_channels):
            chunk[i] = self.amplitudes[i] * np.sin(
                2 * np.pi * self.frequencies[i] * t + self.phases[i]
            )
        
        # Add noise
        chunk += 0.1 * np.random.randn(*chunk.shape)
        
        # Update time
        self.time += self.chunk_duration_ms / 1000
        
        return chunk.astype(np.float32)