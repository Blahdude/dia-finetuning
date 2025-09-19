#!/usr/bin/env python3
"""Test exact duration control using the approach from dia_instrumentals_duration.md"""

import torch
import soundfile as sf
from pathlib import Path
from transformers import LogitsProcessor, LogitsProcessorList

from dia.config import DiaConfig
from dia.model import Dia


def seconds_to_tokens(T_seconds: float, r: float = 86.1328125) -> int:
    """Convert seconds to tokens using calibrated rate."""
    return int(round(r * T_seconds))


class SuppressEOSUntil(LogitsProcessor):
    """Suppress EOS token until target new tokens reached."""
    def __init__(self, eos_id: int, start_len: int, target_new: int):
        self.eos_id = eos_id
        self.start_len = start_len
        self.target_new = target_new
    
    def __call__(self, input_ids, scores):
        new = input_ids.shape[1] - self.start_len
        if new < self.target_new:
            scores[:, self.eos_id] = float("-inf")
        return scores


def test_duration_control():
    """Test generating exact durations of piano trap music."""
    print("Testing exact duration control for piano trap generation...")
    
    # Load config and model
    config_path = Path("dia/music_config.json")
    config = DiaConfig.load(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load trained model
    checkpoint_path = Path("piano_trap_checkpoints/ckpt_epoch100.pth")
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    dia = Dia.from_local(
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        device=device
    )
    
    # Test multiple durations
    test_durations = [3.0, 6.0, 10.0, 15.0]  # seconds
    text_prompt = "piano trap beat"
    
    for duration in test_durations:
        print(f"\nGenerating {duration} seconds of piano trap...")
        
        # Calculate token budget
        target_tokens = seconds_to_tokens(duration)
        print(f"Target tokens for {duration}s: {target_tokens}")
        
        # Check if within position limits
        max_pos = getattr(config.model.decoder, 'max_position_embeddings', 3072)
        if target_tokens > max_pos:
            print(f"WARNING: Target {target_tokens} exceeds max_position_embeddings={max_pos}")
            print("Consider chunked generation for longer sequences")
            continue
        
        # Method 1: Use min/max tokens approach (simpler)
        print("Method 1: Using min_tokens/max_tokens")
        try:
            generated_audio_v1 = dia.generate(
                text=text_prompt,
                max_tokens=target_tokens,  # Maximum tokens
                min_tokens=target_tokens,  # Force exact length by setting min=max
                temperature=1.5,  # Our magic temperature
                top_p=0.8,
                cfg_scale=1.0     # Our magic CFG scale
            )
            
            output_path_v1 = f"duration_test_{duration}s_method1.wav"
            sf.write(output_path_v1, generated_audio_v1, 44100)
            actual_duration_v1 = len(generated_audio_v1) / 44100
            print(f"  Generated: {actual_duration_v1:.2f}s, Target: {duration}s")
            print(f"  Saved to: {output_path_v1}")
            
        except Exception as e:
            print(f"  Method 1 failed: {e}")
        
        # Method 2: Add EOS suppression (more robust)
        print("Method 2: Using EOS suppression")
        try:
            # Note: This would require modifying the Dia.generate() method
            # to accept logits_processor parameter. For now, we'll skip this
            # unless we want to modify the core generation code.
            print("  Skipping Method 2 - would require modifying Dia.generate() method")
            
        except Exception as e:
            print(f"  Method 2 failed: {e}")
    
    print("\nDuration control test completed!")
    print("Check the generated files to verify exact durations")


def calibrate_tokens_per_second():
    """Calibrate the actual tokens/second rate for our model."""
    print("Calibrating tokens per second rate...")
    
    config_path = Path("dia/music_config.json")
    config = DiaConfig.load(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint_path = Path("piano_trap_checkpoints/ckpt_epoch100.pth")
    dia = Dia.from_local(str(config_path), str(checkpoint_path), device)
    
    # Test with known token count
    test_tokens = 860  # Should be ~10 seconds if r=86
    print(f"Generating {test_tokens} tokens...")
    
    generated_audio = dia.generate(
        text="piano trap beat",
        max_tokens=test_tokens,
        min_tokens=test_tokens,  # Force exact length
        temperature=1.5,
        top_p=0.8,
        cfg_scale=1.0
    )
    
    actual_seconds = len(generated_audio) / 44100
    measured_rate = test_tokens / actual_seconds
    
    print(f"Test tokens: {test_tokens}")
    print(f"Actual duration: {actual_seconds:.2f} seconds")
    print(f"Measured rate: {measured_rate:.2f} tokens/second")
    print(f"Theoretical rate: 86.13 tokens/second")
    
    sf.write("calibration_test.wav", generated_audio, 44100)
    print("Saved calibration test to: calibration_test.wav")
    
    return measured_rate


if __name__ == "__main__":
    # First calibrate our tokens/second rate
    measured_rate = calibrate_tokens_per_second()
    
    print(f"\nUsing measured rate: {measured_rate:.2f} tokens/second")
    
    # Then test duration control
    test_duration_control()