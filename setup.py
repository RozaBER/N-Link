from setuptools import setup, find_packages

setup(
    name="n-link",
    version="0.1.0",
    author="N-Link Team",
    description="MEG to Speech/Text System - Real-time MEG signal conversion",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "librosa>=0.9.0",
        "transformers>=4.35.0",
        "mne>=1.0.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "wandb>=0.12.0",
        "einops>=0.6.0",
        "accelerate>=0.21.0",
        "bitsandbytes>=0.41.0",
    ],
)