#!/usr/bin/env python3
"""Test inference with trained sine wave model."""

import torch
import soundfile as sf
from pathlib import Path

from dia.config import DiaConfig
from dia.model import Dia


def test_sine_inference():
    """Test generating sine wave from trained model."""
    print("Testing inference with trained sine wave model...")
    
    # Load config
    config_path = Path("dia/music_config.json")
    config = DiaConfig.load(config_path)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load trained model
    checkpoint_path = Path("sine_checkpoints/ckpt_epoch10.pth")
    if not checkpoint_path.exists():
        # Try alternate path in case it saved to wrong folder
        checkpoint_path = Path("checkpoints_sine/ckpt_epoch10.pth")
    
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Available checkpoints:")
        for p in Path(".").glob("**/ckpt_*.pth"):
            print(f"  {p}")
        return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Create Dia model with trained weights
    dia = Dia.from_local(
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        device=device
    )
    
    # Test inference with the same prompt used in training
    text_prompt = "sine wave"
    print(f"Generating audio for text: '{text_prompt}'")
    
    # Generate audio
    generated_audio = dia.generate(
        text=text_prompt,
        max_tokens=2584,  # Same length as training data
        temperature=0.1,  # Low temperature for deterministic output
        top_p=0.9,
        cfg_scale=3.0
    )
    
    # Save generated audio
    output_path = "generated_sine_wave.wav"
    sf.write(output_path, generated_audio, 44100)
    print(f"Saved generated audio to: {output_path}")
    
    # Load original for comparison
    if Path("test_sine_wave_30s.wav").exists():
        original_audio, _ = sf.read("test_sine_wave_30s.wav")
        sf.write("original_sine_for_comparison.wav", original_audio, 44100)
        print("Saved original sine wave to: original_sine_for_comparison.wav")
    
    print("\nListening test:")
    print("1. Play original_sine_for_comparison.wav (ground truth)")
    print("2. Play generated_sine_wave.wav (model output)")
    print("3. Compare if they sound similar")
    
    # Basic quality check
    print(f"\nGenerated audio stats:")
    print(f"  Length: {len(generated_audio) / 44100:.1f} seconds")
    print(f"  Range: [{generated_audio.min():.3f}, {generated_audio.max():.3f}]")
    print(f"  Shape: {generated_audio.shape}")
    
    return generated_audio


if __name__ == "__main__":
    test_sine_inference()