import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, LlavaForConditionalGeneration
from typing import Optional, Dict, List, Tuple
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal information"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        return x + self.pe[:, :x.size(1)]


class MEGToLLaVAAdapter(nn.Module):
    """
    Adapter to convert MEG brain features to LLaVA visual token space
    Enables MEG signals to be processed as "visual" inputs by LLaVA
    """
    
    def __init__(
        self,
        meg_feature_dim: int = 384,
        llava_visual_dim: int = 576,
        num_visual_tokens: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.meg_feature_dim = meg_feature_dim
        self.llava_visual_dim = llava_visual_dim
        self.num_visual_tokens = num_visual_tokens
        
        # Initial projection from MEG to intermediate dimension
        self.meg_projection = nn.Sequential(
            nn.Linear(meg_feature_dim, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(768, llava_visual_dim),
        )
        
        # Positional encoding for temporal information
        self.positional_encoding = PositionalEncoding(llava_visual_dim)
        
        # Cross-modal alignment with self-attention
        self.cross_modal_align = nn.MultiheadAttention(
            embed_dim=llava_visual_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Learnable visual token queries for compression
        self.visual_queries = nn.Parameter(
            torch.randn(1, num_visual_tokens, llava_visual_dim)
        )
        
        # Cross-attention for token compression
        self.token_compression = nn.MultiheadAttention(
            embed_dim=llava_visual_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(llava_visual_dim)
        self.norm2 = nn.LayerNorm(llava_visual_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(llava_visual_dim, llava_visual_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llava_visual_dim * 4, llava_visual_dim),
        )
        
        # Output projection
        self.output_norm = nn.LayerNorm(llava_visual_dim)
    
    def forward(
        self,
        meg_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert MEG features to LLaVA visual tokens
        
        Args:
            meg_features: (B, T, D) or (B, D) MEG brain features
            attention_mask: (B, T) optional attention mask
            
        Returns:
            Dictionary containing:
                - visual_tokens: (B, num_visual_tokens, llava_visual_dim)
                - attention_weights: attention weights from compression
        """
        # Handle both sequence and pooled inputs
        if meg_features.dim() == 2:
            meg_features = meg_features.unsqueeze(1)  # (B, 1, D)
        
        B, T, D = meg_features.shape
        
        # Project MEG features to visual dimension
        visual_features = self.meg_projection(meg_features)  # (B, T, llava_visual_dim)
        
        # Add positional encoding
        visual_features = self.positional_encoding(visual_features)
        
        # Self-attention for cross-modal alignment
        aligned_features, _ = self.cross_modal_align(
            visual_features,
            visual_features,
            visual_features,
            key_padding_mask=attention_mask if attention_mask is not None else None,
        )
        aligned_features = self.norm1(aligned_features + visual_features)
        
        # Compress to fixed number of visual tokens using cross-attention
        visual_queries = self.visual_queries.expand(B, -1, -1)  # (B, num_tokens, dim)
        
        compressed_tokens, compression_weights = self.token_compression(
            visual_queries,
            aligned_features,
            aligned_features,
            key_padding_mask=attention_mask if attention_mask is not None else None,
        )
        compressed_tokens = self.norm2(compressed_tokens + visual_queries)
        
        # Feed-forward network
        output_tokens = compressed_tokens + self.ffn(compressed_tokens)
        output_tokens = self.output_norm(output_tokens)
        
        return {
            "visual_tokens": output_tokens,
            "attention_weights": compression_weights,
        }
    
    def create_visual_embeddings_for_llava(
        self,
        meg_features: torch.Tensor,
        text_prompt: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Create visual embeddings compatible with LLaVA input format
        
        Args:
            meg_features: MEG brain features
            text_prompt: Optional text prompt to prepend
            
        Returns:
            Dictionary with LLaVA-compatible inputs
        """
        # Get visual tokens
        adapter_output = self.forward(meg_features)
        visual_tokens = adapter_output["visual_tokens"]
        
        # Create placeholder for image features (using MEG features)
        # LLaVA expects image features in specific format
        llava_inputs = {
            "pixel_values": None,  # We bypass image processing
            "image_features": visual_tokens,  # Our MEG-derived "visual" features
        }
        
        if text_prompt:
            llava_inputs["text_prompt"] = text_prompt
        
        return llava_inputs


class MEGLLaVAInterface(nn.Module):
    """
    Complete interface for integrating MEG encoder with LLaVA
    Handles the full pipeline from MEG signals to LLaVA inputs
    """
    
    def __init__(
        self,
        brain_encoder: nn.Module,
        adapter: MEGToLLaVAAdapter,
        llava_model_name: str = "llava-hf/llava-1.5-7b-hf",
        freeze_llava: bool = True,
        use_lora: bool = True,
        lora_rank: int = 8,
    ):
        super().__init__()
        self.brain_encoder = brain_encoder
        self.adapter = adapter
        
        # Load LLaVA model
        self.llava = LlavaForConditionalGeneration.from_pretrained(
            llava_model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        
        # Freeze LLaVA weights if specified
        if freeze_llava:
            for param in self.llava.parameters():
                param.requires_grad = False
        
        # Apply LoRA if specified
        if use_lora and not freeze_llava:
            self._apply_lora(lora_rank)
        
        # Create custom vision projector that accepts our MEG features
        self.vision_projector = nn.Linear(
            self.adapter.llava_visual_dim,
            self.llava.config.text_config.hidden_size,
        )
    
    def _apply_lora(self, rank: int):
        """Apply LoRA to LLaVA language model"""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=rank,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"],
            )
            
            self.llava.language_model = get_peft_model(
                self.llava.language_model,
                peft_config
            )
        except ImportError:
            print("Warning: PEFT not installed. LoRA will not be applied.")
            print("Install with: pip install peft")
    
    def forward(
        self,
        meg_signal: torch.Tensor,
        text_input: Optional[str] = None,
        subject_id: Optional[torch.Tensor] = None,
        generate_kwargs: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass from MEG to LLaVA outputs
        
        Args:
            meg_signal: (B, C, T) MEG signals
            text_input: Optional text prompt
            subject_id: Subject IDs for brain encoder
            generate_kwargs: Generation parameters for LLaVA
            
        Returns:
            Dictionary with model outputs
        """
        # Encode MEG signals
        meg_features = self.brain_encoder(
            meg_signal,
            subject_id=subject_id,
            return_sequence=True
        )
        
        # Convert to visual tokens
        adapter_output = self.adapter(meg_features)
        visual_tokens = adapter_output["visual_tokens"]
        
        # Project to LLaVA text dimension
        visual_embeds = self.vision_projector(visual_tokens)
        
        # Prepare inputs for LLaVA
        outputs = {
            "visual_embeddings": visual_embeds,
            "meg_features": meg_features,
            "adapter_attention": adapter_output["attention_weights"],
        }
        
        # If text input provided, prepare for generation
        if text_input and hasattr(self, 'tokenizer'):
            # This would be used during inference
            # Actual generation code would go here
            pass
        
        return outputs
    
    def prepare_for_training(self):
        """Set up model for training"""
        # Ensure brain encoder and adapter are trainable
        self.brain_encoder.train()
        self.adapter.train()
        
        # Keep LLaVA in eval mode if frozen
        if all(not p.requires_grad for p in self.llava.parameters()):
            self.llava.eval()
    
    def prepare_for_inference(self):
        """Set up model for inference"""
        self.eval()
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.llava, 'gradient_checkpointing_enable'):
            self.llava.gradient_checkpointing_enable()