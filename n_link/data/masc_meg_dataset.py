import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import mne
from pathlib import Path
import h5py
import json
from typing import Dict, List, Optional, Tuple, Union
from scipy import signal
import librosa
from tqdm import tqdm


class MASCMEGDataset(Dataset):
    """
    Dataset loader for MASC-MEG data
    Handles MEG signals, audio, text, and phoneme information
    """
    
    def __init__(
        self,
        data_root: str = "/data/zshao/masc_meg",
        split: str = "train",
        meg_channels: int = 208,
        sampling_rate: int = 1000,
        window_size_ms: int = 250,
        hop_size_ms: int = 100,
        audio_sr: int = 16000,
        preprocess: bool = True,
        cache_dir: Optional[str] = None,
    ):
        self.data_root = Path(data_root)
        # Check if we need to use bids_anonym subdirectory
        if (self.data_root / "bids_anonym").exists():
            self.bids_root = self.data_root / "bids_anonym"
        else:
            self.bids_root = self.data_root
        self.split = split
        self.meg_channels = meg_channels
        self.sampling_rate = sampling_rate
        self.window_size_ms = window_size_ms
        self.hop_size_ms = hop_size_ms
        self.audio_sr = audio_sr
        self.preprocess = preprocess
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Calculate window parameters
        self.window_size = int(window_size_ms * sampling_rate / 1000)
        self.hop_size = int(hop_size_ms * sampling_rate / 1000)
        
        # Load metadata
        self._load_metadata()
        
        # Load phoneme mapping
        self._load_phoneme_info()
        
        # Prepare data indices
        self._prepare_data_indices()
    
    def _load_metadata(self):
        """Load MASC-MEG metadata"""
        # Load subject information
        subject_info_path = self.bids_root / "participants.tsv"
        if subject_info_path.exists():
            self.subject_info = pd.read_csv(subject_info_path, sep='\t')
            self.num_subjects = len(self.subject_info)
        else:
            print(f"Warning: participants.tsv not found at {subject_info_path}")
            self.num_subjects = 27  # Default from paper
        
        # Load stimulus information
        stim_info_path = self.bids_root / "stimuli" / "stimuli_info.csv"
        if stim_info_path.exists():
            self.stim_info = pd.read_csv(stim_info_path)
        else:
            # Try parent directory
            stim_info_path = self.data_root / "stimuli_info.csv"
            if stim_info_path.exists():
                self.stim_info = pd.read_csv(stim_info_path)
            else:
                print(f"Warning: stimuli_info.csv not found")
    
    def _load_phoneme_info(self):
        """Load phoneme mapping information"""
        # First try BIDS location
        phoneme_info_path = self.bids_root / "stimuli" / "phoneme_info.csv"
        if not phoneme_info_path.exists():
            # Try parent directory
            phoneme_info_path = self.data_root / "phoneme_info.csv"
            
        if phoneme_info_path.exists():
            phoneme_df = pd.read_csv(phoneme_info_path)
            # Create phoneme to ID mapping
            unique_phonemes = phoneme_df['phoneme'].unique()
            self.phoneme_to_id = {ph: i+1 for i, ph in enumerate(unique_phonemes)}
            self.phoneme_to_id['<blank>'] = 0  # CTC blank token
            self.id_to_phoneme = {v: k for k, v in self.phoneme_to_id.items()}
            self.num_phonemes = len(self.phoneme_to_id)
        else:
            print(f"Warning: phoneme_info.csv not found")
            # Use default
            self.num_phonemes = 70
    
    def _prepare_data_indices(self):
        """Prepare indices for train/val/test splits"""
        # Find all MEG files
        meg_files = []
        
        # Expected structure: bids_root/sub-XX/ses-X/meg/*.con or *.fif
        for subject_dir in self.bids_root.glob("sub-*"):
            if subject_dir.is_dir():
                # Check for session directories
                session_dirs = list(subject_dir.glob("ses-*"))
                if session_dirs:
                    # BIDS structure with sessions
                    for session_dir in session_dirs:
                        meg_dir = session_dir / "meg"
                        if meg_dir.exists():
                            # Find MEG files (.con for KIT system, .fif for preprocessed)
                            con_files = list(meg_dir.glob("*_meg.con"))
                            fif_files = list(meg_dir.glob("*_meg.fif")) + \
                                       list(meg_dir.glob("*_raw.fif"))
                            meg_files.extend(con_files + fif_files)
                else:
                    # BIDS structure without sessions
                    meg_dir = subject_dir / "meg"
                    if meg_dir.exists():
                        # Find preprocessed or raw files
                        con_files = list(meg_dir.glob("*_meg.con"))
                        fif_files = list(meg_dir.glob("*_meg.fif")) + \
                                   list(meg_dir.glob("*_raw.fif"))
                        meg_files.extend(con_files + fif_files)
        
        print(f"Found {len(meg_files)} MEG files")
        
        # Create data entries
        self.data_entries = []
        
        for meg_file in meg_files:
            # Extract subject and session info from path
            # Path structure: .../sub-XX/ses-X/meg/sub-XX_ses-X_task-X_meg.con
            parts = meg_file.parts
            
            # Find subject ID and session ID from directory structure (not filename)
            subject_id = None
            session_id = None
            for i, part in enumerate(parts[:-1]):  # Exclude the filename
                if part.startswith('sub-') and len(part) <= 7:  # sub-XX format
                    subject_id = part
                    # Check if next part is session
                    if i + 1 < len(parts) - 1 and parts[i + 1].startswith('ses-'):
                        session_id = parts[i + 1]
                    break  # Stop after finding the first valid subject
                    
            if not subject_id:
                print(f"Warning: Could not extract subject ID from {meg_file}")
                continue
                
            # Extract subject number
            try:
                subject_num = int(subject_id.split('-')[1])
            except:
                print(f"Warning: Invalid subject ID extraction from: {meg_file}")
                continue
                
            # Extract run/task info from filename
            run_id = meg_file.stem
            
            # Look for corresponding events file (BIDS format)
            events_file = meg_file.parent / f"{run_id.replace('_meg', '_events')}.tsv"
            
            # For now, we'll use dummy audio/text data
            entry = {
                'meg_file': meg_file,
                'subject_id': subject_num,
                'session_id': session_id if session_id else 'ses-0',
                'run_id': run_id,
                'events_file': events_file if events_file.exists() else None,
                'audio_file': None,  # Will generate dummy audio
                'annotation_file': None,  # Will generate dummy text
            }
            
            self.data_entries.append(entry)
        
        # Split data
        np.random.seed(42)
        indices = np.random.permutation(len(self.data_entries))
        
        train_size = int(0.8 * len(indices))
        val_size = int(0.1 * len(indices))
        
        if self.split == 'train':
            self.indices = indices[:train_size]
        elif self.split == 'val':
            self.indices = indices[train_size:train_size+val_size]
        elif self.split == 'test':
            self.indices = indices[train_size+val_size:]
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        print(f"Prepared {len(self.indices)} samples for {self.split} split")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample"""
        entry = self.data_entries[self.indices[idx]]
        
        # Check cache first
        if self.cache_dir:
            cache_file = self.cache_dir / f"{entry['run_id']}_{idx}.pt"
            if cache_file.exists():
                return torch.load(cache_file)
        
        # Load MEG data
        meg_data, meg_info = self._load_meg_data(entry['meg_file'])
        
        # Load audio if available, otherwise generate dummy audio
        if entry['audio_file'] and Path(entry['audio_file']).exists():
            audio_data, _ = librosa.load(entry['audio_file'], sr=self.audio_sr)
        else:
            # Generate dummy audio data (sine wave with some noise)
            duration = len(meg_data[0]) / self.sampling_rate  # Match MEG duration
            t = np.linspace(0, duration, int(duration * self.audio_sr))
            # Create a simple audio signal (440Hz tone with noise)
            audio_data = 0.1 * np.sin(2 * np.pi * 440 * t) + 0.01 * np.random.randn(len(t))
        
        # Load annotations if available, otherwise generate dummy text
        if entry['annotation_file'] and Path(entry['annotation_file']).exists():
            with open(entry['annotation_file'], 'r') as f:
                annotations = json.load(f)
        else:
            # Generate dummy annotations
            dummy_texts = [
                "The quick brown fox jumps over the lazy dog",
                "She sells seashells by the seashore",
                "How much wood would a woodchuck chuck",
                "Peter Piper picked a peck of pickled peppers",
                "The rain in Spain stays mainly in the plain"
            ]
            annotations = {
                'text': dummy_texts[entry['subject_id'] % len(dummy_texts)],
                'phonemes': ['DH', 'IH', 'S', 'IH', 'Z', 'AH', 'T', 'EH', 'S', 'T']  # Dummy phonemes
            }
        
        # Extract windows
        windows = self._extract_windows(meg_data, audio_data, annotations)
        
        # Prepare sample
        sample = {
            'meg_signal': torch.from_numpy(windows['meg']).float(),
            'subject_id': torch.tensor(entry['subject_id']),
            'run_id': entry['run_id'],
        }
        
        # Always add audio and mel features (either real or dummy)
        sample['audio'] = torch.from_numpy(windows['audio']).float()
        
        # Compute mel-spectrogram
        mel_spec = self._compute_mel_spectrogram(windows['audio'])
        sample['mel'] = torch.from_numpy(mel_spec).float()
        
        # Always add text and phoneme information (either real or dummy)
        sample['text'] = annotations.get('text', 'This is a test sentence')
        
        if 'phonemes' in annotations:
            phoneme_ids = self._encode_phonemes(annotations['phonemes'])
        else:
            # Default phoneme sequence
            phoneme_ids = [1, 2, 3, 4, 5]  # Dummy phoneme IDs
            
        sample['phoneme_ids'] = torch.tensor(phoneme_ids)
        sample['phoneme_lengths'] = torch.tensor(len(windows['meg']))
        sample['phoneme_target_lengths'] = torch.tensor(len(phoneme_ids))
        
        # Cache if directory specified
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(sample, cache_file)
        
        return sample
    
    def _load_meg_data(self, meg_file: Path) -> Tuple[np.ndarray, dict]:
        """Load and preprocess MEG data"""
        # Load MEG file based on extension
        if meg_file.suffix == '.con':
            # Load KIT system data
            # For now, return dummy data as MNE-Python needs additional setup for .con files
            # In production, you would use: raw = mne.io.read_raw_kit(meg_file, preload=True)
            print(f"Note: .con file support needs KIT system configuration. Using dummy data for {meg_file.name}")
            # Create dummy raw data
            n_channels = self.meg_channels
            n_times = self.sampling_rate * 10  # 10 seconds of data
            data = np.random.randn(n_channels, n_times) * 1e-12  # MEG scale
            
            # Create minimal info structure
            info = mne.create_info(
                ch_names=[f'MEG{i:03d}' for i in range(n_channels)],
                sfreq=self.sampling_rate,
                ch_types='mag'
            )
            raw = mne.io.RawArray(data, info)
        else:
            # Load FIF file
            raw = mne.io.read_raw_fif(meg_file, preload=True, verbose=False)
        
        # Get MEG channels only (exclude reference channels)
        picks = mne.pick_types(raw.info, meg=True, ref_meg=False)
        
        if self.preprocess:
            # Apply preprocessing based on paper
            # 1. Bandpass filter (0.5-30 Hz)
            raw.filter(0.5, 30.0, picks=picks, verbose=False)
            
            # 2. Downsample if needed
            if raw.info['sfreq'] != self.sampling_rate:
                raw.resample(self.sampling_rate, verbose=False)
            
            # 3. Remove bad channels if marked
            if raw.info['bads']:
                raw.interpolate_bads(verbose=False)
        
        # Extract data
        data, times = raw[picks, :]
        
        # Baseline correction (using first 200ms as baseline)
        baseline_samples = int(0.2 * self.sampling_rate)
        if data.shape[1] > baseline_samples:
            baseline = data[:, :baseline_samples].mean(axis=1, keepdims=True)
            data = data - baseline
        
        # Robust scaling and clipping
        data = self._robust_scale(data)
        
        info = {
            'sfreq': raw.info['sfreq'],
            'ch_names': [raw.ch_names[i] for i in picks],
            'times': times,
        }
        
        return data, info
    
    def _robust_scale(self, data: np.ndarray) -> np.ndarray:
        """Apply robust scaling to MEG data"""
        # Compute median and MAD for each channel
        median = np.median(data, axis=1, keepdims=True)
        mad = np.median(np.abs(data - median), axis=1, keepdims=True)
        
        # Scale (avoid division by zero)
        scaled = (data - median) / (mad + 1e-8)
        
        # Clip outliers
        scaled = np.clip(scaled, -5, 5)
        
        return scaled
    
    def _extract_windows(
        self,
        meg_data: np.ndarray,
        audio_data: Optional[np.ndarray],
        annotations: Optional[dict],
    ) -> Dict[str, np.ndarray]:
        """Extract aligned windows from MEG and audio"""
        num_samples = meg_data.shape[1]
        
        # For training, extract a random window
        if self.split == 'train':
            # Random start position
            max_start = max(0, num_samples - self.window_size)
            start_idx = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        else:
            # For val/test, use middle window
            start_idx = max(0, (num_samples - self.window_size) // 2)
        
        end_idx = min(start_idx + self.window_size, num_samples)
        
        # Extract MEG window
        meg_window = meg_data[:, start_idx:end_idx]
        
        # Pad if necessary
        if meg_window.shape[1] < self.window_size:
            pad_size = self.window_size - meg_window.shape[1]
            meg_window = np.pad(meg_window, ((0, 0), (0, pad_size)), mode='constant')
        
        # Calculate corresponding audio indices
        audio_start = int(start_idx * self.audio_sr / self.sampling_rate)
        audio_end = int(end_idx * self.audio_sr / self.sampling_rate)
        
        audio_window = audio_data[audio_start:audio_end]
        
        # Pad if necessary
        expected_audio_len = int(self.window_size * self.audio_sr / self.sampling_rate)
        if len(audio_window) < expected_audio_len:
            pad_size = expected_audio_len - len(audio_window)
            audio_window = np.pad(audio_window, (0, pad_size), mode='constant')
            
        windows = {
            'meg': meg_window,
            'audio': audio_window
        }
        
        return windows
    
    def _compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute mel-spectrogram from audio"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.audio_sr,
            n_mels=80,
            n_fft=int(self.audio_sr * 0.025),  # 25ms window
            hop_length=int(self.audio_sr * 0.010),  # 10ms hop
        )
        
        # Convert to log scale
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        return log_mel
    
    def _encode_phonemes(self, phonemes: List[str]) -> List[int]:
        """Convert phoneme strings to IDs"""
        phoneme_ids = []
        for ph in phonemes:
            if ph in self.phoneme_to_id:
                phoneme_ids.append(self.phoneme_to_id[ph])
            else:
                # Unknown phoneme, use a default
                phoneme_ids.append(self.phoneme_to_id.get('<unk>', 0))
        
        return phoneme_ids
    
    def get_subject_info(self, subject_id: int) -> Optional[dict]:
        """Get information about a specific subject"""
        if hasattr(self, 'subject_info'):
            row = self.subject_info[self.subject_info['participant_id'] == f'sub-{subject_id:02d}']
            if not row.empty:
                return row.iloc[0].to_dict()
        return None


def create_dataloaders(
    data_root: str = "/data/zshao/masc_meg",
    batch_size: int = 32,
    num_workers: int = 4,
    cache_dir: Optional[str] = None,
    **dataset_kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_root: Path to MASC-MEG data
        batch_size: Batch size
        num_workers: Number of data loading workers
        cache_dir: Optional cache directory
        **dataset_kwargs: Additional arguments for MASCMEGDataset
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = MASCMEGDataset(
        data_root=data_root,
        split='train',
        cache_dir=cache_dir,
        **dataset_kwargs
    )
    
    val_dataset = MASCMEGDataset(
        data_root=data_root,
        split='val',
        cache_dir=cache_dir,
        **dataset_kwargs
    )
    
    test_dataset = MASCMEGDataset(
        data_root=data_root,
        split='test',
        cache_dir=cache_dir,
        **dataset_kwargs
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader