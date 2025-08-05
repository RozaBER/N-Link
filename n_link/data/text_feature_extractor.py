import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, CLIPTextModel, CLIPTokenizer
from typing import Dict, List, Optional, Union, Tuple
import numpy as np


class TextFeatureExtractor:
    """
    Extract text features for brain-to-text alignment training
    Supports both CLIP and sentence transformer models
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cuda",
        max_length: int = 77,
        use_clip: bool = True,
    ):
        self.device = device
        self.max_length = max_length
        self.use_clip = use_clip
        
        if use_clip:
            # Load CLIP text encoder
            self.model = CLIPTextModel.from_pretrained(model_name).to(device)
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        else:
            # Load sentence transformer
            self.model = AutoModel.from_pretrained(model_name).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.model.eval()
    
    @torch.no_grad()
    def extract_features(
        self,
        texts: Union[str, List[str]],
        return_pooled: bool = True,
    ) -> torch.Tensor:
        """
        Extract text features
        
        Args:
            texts: Single text or list of texts
            return_pooled: Return pooled features (True) or sequence features (False)
            
        Returns:
            Text features tensor
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        
        # Extract features
        if self.use_clip:
            outputs = self.model(**inputs)
            if return_pooled:
                features = outputs.pooler_output
            else:
                features = outputs.last_hidden_state
        else:
            outputs = self.model(**inputs)
            if return_pooled:
                # Mean pooling
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(
                    token_embeddings.size()
                ).float()
                features = torch.sum(
                    token_embeddings * input_mask_expanded, 1
                ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            else:
                features = outputs.last_hidden_state
        
        return features
    
    def batch_extract(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Extract features for large list of texts in batches
        
        Args:
            texts: List of texts
            batch_size: Batch size for processing
            show_progress: Show progress bar
            
        Returns:
            Stacked feature tensor
        """
        features = []
        
        if show_progress:
            from tqdm import tqdm
            pbar = tqdm(total=len(texts), desc="Extracting text features")
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_features = self.extract_features(batch_texts)
            features.append(batch_features.cpu())
            
            if show_progress:
                pbar.update(len(batch_texts))
        
        if show_progress:
            pbar.close()
        
        return torch.cat(features, dim=0)


class WordPieceAligner:
    """
    Align word-level annotations with subword tokens
    Useful for fine-grained text-brain alignment
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def align_words_to_tokens(
        self,
        text: str,
        words: List[str],
        word_timestamps: List[Tuple[float, float]],
    ) -> Dict[str, List]:
        """
        Align word-level timestamps to subword tokens
        
        Args:
            text: Full text
            words: List of words
            word_timestamps: List of (start, end) timestamps for each word
            
        Returns:
            Dictionary with aligned tokens and timestamps
        """
        # Tokenize full text
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        
        tokens = encoding['input_ids'][0]
        offsets = encoding['offset_mapping'][0]
        
        # Align words to tokens
        token_timestamps = []
        current_word_idx = 0
        current_char_idx = 0
        
        for i, (start_offset, end_offset) in enumerate(offsets):
            if start_offset == end_offset:  # Special token
                token_timestamps.append((-1, -1))
                continue
            
            # Find which word this token belongs to
            while current_word_idx < len(words):
                word = words[current_word_idx]
                word_start = text.find(word, current_char_idx)
                word_end = word_start + len(word)
                
                if start_offset >= word_start and end_offset <= word_end:
                    # Token is within this word
                    token_timestamps.append(word_timestamps[current_word_idx])
                    break
                elif start_offset >= word_end:
                    # Move to next word
                    current_word_idx += 1
                    current_char_idx = word_end
                else:
                    # Partial overlap or gap
                    token_timestamps.append(word_timestamps[current_word_idx])
                    break
            else:
                # No more words
                token_timestamps.append((-1, -1))
        
        return {
            'tokens': tokens,
            'token_timestamps': token_timestamps,
            'offset_mapping': offsets,
        }


def augment_text_for_training(
    text: str,
    augmentation_prob: float = 0.3,
) -> str:
    """
    Apply text augmentation for robust training
    
    Args:
        text: Input text
        augmentation_prob: Probability of applying augmentation
        
    Returns:
        Augmented text
    """
    import random
    
    if random.random() > augmentation_prob:
        return text
    
    augmentation_type = random.choice(['synonym', 'deletion', 'swap'])
    
    words = text.split()
    
    if augmentation_type == 'synonym':
        # Simple synonym replacement (in practice, use WordNet or similar)
        replacements = {
            'said': 'stated',
            'big': 'large',
            'small': 'tiny',
            'fast': 'quick',
            'good': 'great',
        }
        
        for i, word in enumerate(words):
            if word.lower() in replacements and random.random() < 0.3:
                words[i] = replacements[word.lower()]
    
    elif augmentation_type == 'deletion':
        # Random word deletion
        if len(words) > 3:
            num_delete = max(1, int(len(words) * 0.1))
            indices = random.sample(range(len(words)), num_delete)
            words = [w for i, w in enumerate(words) if i not in indices]
    
    elif augmentation_type == 'swap':
        # Random word swap
        if len(words) > 2:
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
    
    return ' '.join(words)