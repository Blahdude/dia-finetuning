#!/usr/bin/env python3
"""Generate a test sine wave for music generation experiments."""

import numpy as np
import soundfile as sf
from pathlib import Path


def generate_sine_wave(
    frequency: float = 440.0,
    duration: float = 30.0,
    sample_rate: int = 44100,
    amplitude: float = 0.5
) -> np.ndarray:
    """Generate a sine wave.
    
    Args:
        frequency: Frequency in Hz (default 440Hz = A4)
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        amplitude: Amplitude (0.0 to 1.0)
        
    Returns:
        Audio array of shape (samples,)
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return sine_wave.astype(np.float32)


def main():
    # Generate 30-second sine wave
    print("Generating 30-second sine wave at 440Hz...")
    audio = generate_sine_wave(
        frequency=440.0,
        duration=30.0,
        sample_rate=44100,
        amplitude=0.5
    )
    
    # Save to file
    output_path = Path("test_sine_wave_30s.wav")
    sf.write(output_path, audio, 44100)
    print(f"Saved sine wave to: {output_path}")
    print(f"Duration: {len(audio) / 44100:.1f} seconds")
    print(f"Shape: {audio.shape}")
    print(f"Sample rate: 44100 Hz")
    

if __name__ == "__main__":
    main()