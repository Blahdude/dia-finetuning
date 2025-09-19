#!/usr/bin/env python3
"""Training script for piano trap beat overfitting."""

import torch
import dac
from pathlib import Path

from dia.config import DiaConfig
from dia.layers import DiaModel
from dia.sine_dataset import SineWaveDataset
from dia.finetune import TrainConfig, train


class PianoTrapDataset(SineWaveDataset):
    """Dataset for piano trap beat - same structure as SineWaveDataset."""
    pass


def main():
    print("Setting up piano trap beat overfitting...")
    
    # Load config
    config_path = Path("dia/music_config.json")
    config = DiaConfig.load(config_path)
    print(f"Loaded config from: {config_path}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load DAC model
    print("Loading DAC model...")
    dac_model = dac.DAC.load(dac.utils.download()).to(device)
    
    # Check if piano trap file exists
    audio_path = Path("instrumental.wav")
    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        print("Please make sure 'instrumental.wav' is in the current directory")
        return
    
    print(f"Using audio file: {audio_path}")
    
    # Create dataset for piano trap beat
    print("Creating piano trap dataset...")
    dataset = PianoTrapDataset(
        audio_path=audio_path, 
        text="piano trap beat", 
        config=config, 
        dac_model=dac_model
    )
    
    # Load model
    print("Loading Dia model...")
    model = DiaModel(config)
    
    # Load pretrained weights
    try:
        from huggingface_hub import hf_hub_download
        ckpt_file = hf_hub_download("nari-labs/Dia-1.6B", filename="dia-v0_1.pth")
        state_dict = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(state_dict)
        print("Loaded pretrained weights")
    except Exception as e:
        print(f"Could not load pretrained weights: {e}")
        print("Training from scratch...")
    
    # Setup training config for piano trap overfitting
    train_config = TrainConfig(
        epochs=300,  # Much longer training for complex music
        batch_size=1,  # Single sample overfitting
        grad_accum_steps=1,
        learning_rate=5e-5,  # Slightly lower LR for stability
        warmup_steps=50,
        eval_step=9999,  # Skip evaluation
        save_step=9999,  # No intermediate saves
        split_ratio=0.0,  # No validation split
        run_name="piano_trap_overfit",
        output_dir=Path("./piano_trap_checkpoints"),
        seed=42
    )
    
    print("Starting training...")
    print(f"  Audio file: {audio_path}")
    print(f"  Text prompt: 'piano trap beat'")
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Batch size: {train_config.batch_size}")
    print(f"  Epochs: {train_config.epochs}")
    print(f"  Learning rate: {train_config.learning_rate}")
    print(f"  Audio length in config: {config.data.audio_length}")
    print(f"  Will save final checkpoint to: {train_config.output_dir}/ckpt_epoch{train_config.epochs}.pth")
    
    # Start training
    train(model, config, dac_model, dataset, train_config)
    print("Training completed!")
    print(f"Run inference with: python test_piano_inference.py")


if __name__ == "__main__":
    main()