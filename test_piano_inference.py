#!/usr/bin/env python3
"""Test inference with trained piano trap model."""

import torch
import soundfile as sf
from pathlib import Path

from dia.config import DiaConfig
from dia.model import Dia


def seconds_to_tokens(T_seconds: float, r: float = 86.1328125) -> int:
    """Convert seconds to tokens using calibrated rate."""
    return int(round(r * T_seconds))


def test_piano_trap_inference(duration_seconds: float = 6.0):
    """Test generating piano trap beat from trained model.
    
    Args:
        duration_seconds: How many seconds of audio to generate
    """
    print("Testing inference with trained piano trap model...")
    print(f"Target duration: {duration_seconds} seconds")
    
    # Load config
    config_path = Path("dia/music_config.json")
    config = DiaConfig.load(config_path)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load trained model
    checkpoint_path = Path("piano_trap_checkpoints/ckpt_epoch100.pth")
    
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
    
    # Test inference with the training prompt
    text_prompt = "piano trap beat"
    print(f"Generating audio for text: '{text_prompt}'")
    
    # Calculate target duration and tokens
    target_tokens = seconds_to_tokens(duration_seconds)
    max_tokens = min(target_tokens, config.data.audio_length)
    
    if target_tokens > config.data.audio_length:
        actual_duration = config.data.audio_length / 86.1328125
        print(f"WARNING: Requested {duration_seconds}s exceeds max length")
        print(f"Generating maximum length: {actual_duration:.1f}s instead")
    
    print(f"Target tokens: {target_tokens}")
    print(f"Using max_tokens: {max_tokens}")
    
    # Generate audio letting model decide natural length
    generated_audio = dia.generate(
        text=text_prompt,
        max_tokens=max_tokens,
        # min_tokens=max_tokens,  # Don't force exact length - let model stop naturally
        temperature=1.5,  # Our magic temperature
        top_p=0.8,        
        cfg_scale=1.0     # Our magic CFG scale
    )
    
    # Save generated audio
    output_path = "generated_piano_trap.wav"
    sf.write(output_path, generated_audio, 44100)
    
    actual_duration = len(generated_audio) / 44100
    print(f"Saved generated audio to: {output_path}")
    print(f"Target duration: {duration_seconds}s, Actual duration: {actual_duration:.2f}s")
    print(f"Duration accuracy: {(actual_duration/duration_seconds)*100:.1f}%")
    
    # Load original for comparison
    if Path("instrumental.wav").exists():
        original_audio, _ = sf.read("instrumental.wav")
        # Convert to mono if stereo for comparison
        if original_audio.ndim > 1:
            original_audio = original_audio.mean(axis=1)
        sf.write("original_piano_trap_for_comparison.wav", original_audio, 44100)
        print("Saved original (mono) to: original_piano_trap_for_comparison.wav")
    
    print("\nListening test:")
    print("1. Play original_piano_trap_for_comparison.wav (ground truth)")
    print("2. Play generated_piano_trap.wav (model output)")
    print("3. Compare if they sound similar")
    
    # Basic quality check
    print(f"\nGenerated audio stats:")
    print(f"  Length: {actual_duration:.2f} seconds")
    print(f"  Range: [{generated_audio.min():.3f}, {generated_audio.max():.3f}]")
    print(f"  Shape: {generated_audio.shape}")
    
    return generated_audio


if __name__ == "__main__":
    import sys
    
    # Allow user to specify duration from command line
    if len(sys.argv) > 1:
        try:
            duration = float(sys.argv[1])
            print(f"Generating {duration} seconds of piano trap beat...")
            test_piano_trap_inference(duration)
        except ValueError:
            print("Error: Duration must be a number")
            print("Usage: python test_piano_inference.py [duration_in_seconds]")
            print("Example: python test_piano_inference.py 10.5")
    else:
        # Default duration
        print("Usage: python test_piano_inference.py [duration_in_seconds]")
        print("Example: python test_piano_inference.py 10.5")
        print("Using default duration of 6 seconds...")
        test_piano_trap_inference()