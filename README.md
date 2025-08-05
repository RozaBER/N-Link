# N-Link Project Introduction

## Key Points

1. **Multi-Stage Training Architecture**: The system uses a progressive training approach:
   - Stage 1: MEG-Mel alignment using contrastive learning
   - Stage 2: Brain-to-text alignment with pre-trained text features
   - Stage 3: End-to-end fine-tuning with all modalities

2. **Real-Time Processing**: Designed for <100ms latency with streaming capabilities, using 250ms buffers with 50ms overlap for continuous MEG signal processing.

3. **Multimodal Output**: Simultaneously generates:
   - Text transcription via LLaVA language model
   - Phoneme sequences using CTC decoding
   - Audio synthesis through HiFi-GAN vocoder

4. **Subject-Adaptive Encoding**: Handles 27 different subjects with learnable subject embeddings for personalized brain signal decoding.

## Technical Architecture

The system consists of four main components:

1. **MEG Brain Encoder**: Processes 208 MEG channels at 1000Hz with spatial attention, multi-scale temporal convolutions, and frequency analysis
2. **MEG-to-LLaVA Adapter**: Bridges brain signals to the visual token space of LLaVA v1.5 (7B parameters)
3. **Multi-Output Decoder**: Generates text, phonemes, and audio from LLaVA hidden states
4. **MEG-Mel Aligner**: Pre-training component for learning brain-audio correspondences

## Dataset

The project uses the MASC-MEG dataset located at `/data/zshao/masc_meg`, which contains:

- 27 subjects with multiple recording sessions
- 4 narrative listening tasks per session
- MEG recordings at 1000Hz with 208+16 channels
- Synchronized audio and text transcriptions

## Training Guide

### Training Process

#### Option 1: Full Pipeline (Recommended)

Run all three stages sequentially:

```bash
python train.py --stage all \
    --data_root /data/zshao/masc_meg \
    --batch_size 32 \
    --use_wandb \
    --wandb_project n-link \
    --checkpoint_dir ./checkpoints
```

#### Option 2: Stage-by-Stage Training

**Stage 1 - MEG-Mel Alignment (50 epochs):**

```bash
python train.py --stage stage1 \
    --data_root /data/zshao/masc_meg \
    --epochs_stage1 50 \
    --lr_stage1 1e-3 \
    --batch_size 64 \
    --contrastive_weight 1.0
```

**Stage 2 - Brain-to-Text Alignment (30 epochs):**

```bash
python train.py --stage stage2 \
    --data_root /data/zshao/masc_meg \
    --epochs_stage2 30 \
    --lr_stage2 5e-4 \
    --batch_size 32 \
    --text_weight 1.0 \
    --alignment_weight 0.1
```

**Stage 3 - End-to-End Fine-tuning (50 epochs):**

```bash
python train.py --stage stage3 \
    --data_root /data/zshao/masc_meg \
    --epochs_stage3 50 \
    --lr_stage3 1e-4 \
    --batch_size 16 \
    --text_weight 1.0 \
    --phoneme_weight 0.3 \
    --audio_weight 0.2 \
    --freeze_llava \
    --use_lora
```

### Key Training Parameters

- **Batch Sizes**: 64 (stage1), 32 (stage2), 16 (stage3) - adjust based on GPU memory
- **Learning Rates**: 1e-3 (stage1), 5e-4 (stage2), 1e-4 (stage3)
- **Loss Weights**:
  - `contrastive_weight`: 1.0 (stage 1)
  - `text_weight`: 1.0 (stages 2-3)
  - `phoneme_weight`: 0.3 (stage 3)
  - `audio_weight`: 0.2 (stage 3)
  - `alignment_weight`: 0.1 (stages 2-3)

### Monitoring Training

**Checkpoint Management:**
- Checkpoints saved to: `./checkpoints/stage{1,2,3}_<timestamp>/`
- Best models: `stage{1,2,3}_best.pt`
- Config saved: `config.json`
