#!/usr/bin/env python3
"""Test training script for sine wave overfitting."""

import torch
import dac
from pathlib import Path

from dia.config import DiaConfig
from dia.layers import DiaModel
from dia.sine_dataset import SineWaveDataset, create_sine_dataset
from dia.finetune import TrainConfig, train, collate_fn


def main():
    print("Setting up sine wave overfitting test...")
    
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
    
    # Create or use existing sine wave
    sine_path = Path("test_sine_wave_30s.wav")
    if not sine_path.exists():
        print("Creating sine wave...")
        sine_path = create_sine_dataset(
            frequency=440.0,
            duration=30.0,
            text="sine wave",
            output_path=sine_path
        )
    else:
        print(f"Using existing sine wave: {sine_path}")
    
    # Create dataset
    print("Creating sine wave dataset...")
    dataset = SineWaveDataset(sine_path, "sine wave", config, dac_model)
    
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
    
    # Setup training config for overfitting
    train_config = TrainConfig(
        epochs=10,
        batch_size=1,  # Small batch for overfitting
        grad_accum_steps=1,
        learning_rate=1e-4,
        warmup_steps=10,
        eval_step=10,
        save_step=9999,  # Very high number so it never saves during training
        split_ratio=0.0,  # No validation split for single sample
        run_name="sine_overfit",
        output_dir=Path("./sine_checkpoints"),  # Different folder
        seed=42
    )
    
    print("Starting training...")
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Batch size: {train_config.batch_size}")
    print(f"  Epochs: {train_config.epochs}")
    print(f"  Audio length in config: {config.data.audio_length}")
    
    # Start training
    train(model, config, dac_model, dataset, train_config)
    print("Training completed!")


if __name__ == "__main__":
    main()