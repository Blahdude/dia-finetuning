#!/usr/bin/env python3
"""Train Dia model from scratch on audio-text pairs."""

import argparse
import torch
import dac
import soundfile as sf
import wandb
from pathlib import Path
from torch.utils.data import Dataset

from dia.config import DiaConfig
from dia.layers import DiaModel
from dia.finetune import TrainConfig, train


class UniversalAudioDataset(Dataset):
    """Universal dataset for any audio file with text description."""
    
    def __init__(self, audio_path: Path, text: str, config: DiaConfig, dac_model: dac.DAC):
        """
        Args:
            audio_path: Path to the audio file
            text: Text description
            config: Dia configuration
            dac_model: Pre-loaded DAC model for encoding
        """
        self.audio_path = audio_path
        self.text = text
        self.config = config
        self.dac_model = dac_model
        
        # Pre-encode the audio once
        self._encode_audio()
        
    def _encode_audio(self):
        """Encode the audio using DAC."""
        # Load audio
        audio_data, sr = sf.read(self.audio_path)
        
        # Convert to tensor with proper shape for DAC
        if audio_data.ndim == 1:
            # Mono audio
            waveform = torch.from_numpy(audio_data).float().unsqueeze(0).unsqueeze(0)
        else:
            # Stereo audio - convert to mono by averaging channels
            mono_audio = audio_data.mean(axis=1)
            waveform = torch.from_numpy(mono_audio).float().unsqueeze(0).unsqueeze(0)
        
        print(f"Loading audio: {self.audio_path}")
        print(f"Audio shape: {waveform.shape}, sample rate: {sr}")
        print(f"Duration: {audio_data.shape[0] / sr:.2f} seconds")
        
        # Move to same device as DAC model
        device = next(self.dac_model.parameters()).device
        waveform = waveform.to(device)
        
        # Encode with DAC
        with torch.no_grad():
            audio_tensor = self.dac_model.preprocess(waveform, sr)
            _, codes, *_ = self.dac_model.encode(audio_tensor, n_quantizers=None)
            
            # Convert to format expected by training: (time_steps, channels)
            self.encoded_audio = codes.squeeze(0).transpose(0, 1).cpu()  # (time_steps, 9)
            
        print(f"Encoded audio shape: {self.encoded_audio.shape}")
        print(f"Encoded audio range: {self.encoded_audio.min().item()} to {self.encoded_audio.max().item()}")
        
        # Check if length matches config
        max_audio_length = self.config.data.audio_length
        actual_length = self.encoded_audio.shape[0]
        
        if actual_length > max_audio_length:
            print(f"WARNING: Audio length {actual_length} > max_audio_length {max_audio_length}")
            print(f"Truncating to fit model...")
            self.encoded_audio = self.encoded_audio[:max_audio_length]
        elif actual_length < max_audio_length:
            print(f"INFO: Audio length {actual_length} < max_audio_length {max_audio_length}")
            print(f"This is fine - model will learn the actual length")
        
        # Estimate duration in seconds
        estimated_duration = actual_length / 86.13  # tokens per second
        print(f"Estimated duration from tokens: {estimated_duration:.2f} seconds")
        
    def __len__(self) -> int:
        # Single sample for overfitting
        return 1
        
    def __getitem__(self, idx: int):
        """Return the audio sample."""
        # Return: (text, encoded_audio, waveform_placeholder)
        waveform_placeholder = torch.zeros(1, 1000)  # Dummy waveform
        return self.text, self.encoded_audio, waveform_placeholder


def get_training_config(epochs: int = 1000, learning_rate: float = 5e-4, output_dir: str = "scratch_checkpoints"):
    """Get training configuration for training from scratch with proper warmup."""
    return TrainConfig(
        epochs=epochs,
        batch_size=1,               # Single sample overfitting
        grad_accum_steps=1,         # Single sample needs immediate updates
        learning_rate=learning_rate, # Conservative LR for scratch training
        warmup_steps=max(epochs // 3, 300),  # Long warmup essential for scratch training
        eval_step=50,               # Check progress frequently
        save_step=50,               # Save checkpoints regularly
        split_ratio=0.0,            # No validation split for overfitting
        run_name=f"scratch_overfit_{Path(output_dir).name}",
        output_dir=Path(output_dir),
        seed=42
    )


def main():
    parser = argparse.ArgumentParser(description="Train Dia model from scratch on audio-text pair")
    parser.add_argument("audio_file", type=str, help="Path to audio file")
    parser.add_argument("text", type=str, help="Text description of the audio")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate (conservative for scratch training)")
    parser.add_argument("--output_dir", type=str, default="scratch_checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--config", type=str, default="dia/music_config.json", help="Path to model config")
    
    args = parser.parse_args()
    
    print(f"Training Dia model FROM SCRATCH on audio-text pair:")
    print(f"  Audio: {args.audio_file}")
    print(f"  Text: '{args.text}'")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Output: {args.output_dir}")
    print(f"  WARNING: Training from scratch - no pretrained weights!")
    
    # Load config
    config_path = Path(args.config)
    config = DiaConfig.load(config_path)
    print(f"Loaded config from: {config_path}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load DAC model
    print("Loading DAC model...")
    dac_model = dac.DAC.load(dac.utils.download()).to(device)
    
    # Check if audio file exists
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        return 1
    
    # Create dataset
    print("Creating dataset...")
    dataset = UniversalAudioDataset(
        audio_path=audio_path,
        text=args.text,
        config=config,
        dac_model=dac_model
    )
    
    # Load model WITHOUT pretrained weights
    print("Creating Dia model from scratch...")
    model = DiaModel(config)
    print("Model initialized with random weights - no pretrained loading!")
    
    # Get training config
    train_config = get_training_config(
        epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output_dir
    )
    
    print("\\nTraining configuration:")
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Batch size: {train_config.batch_size}")
    print(f"  Epochs: {train_config.epochs}")
    print(f"  Learning rate: {train_config.learning_rate}")
    print(f"  Warmup steps: {train_config.warmup_steps}")
    print(f"  Audio length in config: {config.data.audio_length}")
    print(f"  Will save checkpoints to: {train_config.output_dir}")
    
    # Create output directory
    train_config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize W&B
    wandb.init(
        project="dia-music-generation",
        name=f"scratch-30s-lr{args.lr}",
        config={
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "audio_length": config.data.audio_length,
            "sequence_tokens": dataset.encoded_audio.shape[0],
            "batch_size": train_config.batch_size,
            "grad_accum_steps": train_config.grad_accum_steps,
            "audio_file": args.audio_file,
            "text_prompt": args.text,
            "training_type": "from_scratch"
        },
        sync_tensorboard=True
    )
    wandb.watch(model, log_freq=50)
    
    # Start training
    print("\\nStarting training FROM SCRATCH...")
    train(model, config, dac_model, dataset, train_config)
    
    print("\\nTraining completed!")
    print(f"Final checkpoint saved to: {train_config.output_dir}/ckpt_epoch{train_config.epochs}.pth")
    print(f"Run inference with: python inference.py '{args.text}' {train_config.output_dir} --duration 30")
    
    return 0


if __name__ == "__main__":
    exit(main())