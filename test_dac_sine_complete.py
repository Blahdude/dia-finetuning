#!/usr/bin/env python3
"""Complete DAC test - encode and decode sine wave to verify quality."""

import torch
import torchaudio
import soundfile as sf
import numpy as np
from pathlib import Path
import dac


def test_dac_reconstruction():
    """Test complete DAC encode/decode cycle on sine wave."""
    print("Loading DAC model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dac_model = dac.DAC.load(dac.utils.download()).to(device)
    
    print("Loading sine wave...")
    audio_path = Path("test_sine_wave_30s.wav")
    waveform, sr = torchaudio.load(audio_path)
    print(f"Original audio shape: {waveform.shape}, sample rate: {sr}")
    
    # Ensure proper format: (batch_size, channels, sequence_length)
    if waveform.dim() == 2 and waveform.shape[0] == 1:
        # (1, samples) -> (1, 1, samples) for mono
        waveform = waveform.unsqueeze(1)
    
    print(f"Formatted waveform shape: {waveform.shape}")
    waveform = waveform.to(device)
    
    print("Encoding with DAC...")
    with torch.no_grad():
        # Preprocess
        audio_tensor = dac_model.preprocess(waveform, sr).to(device)
        print(f"Preprocessed shape: {audio_tensor.shape}")
        
        # Encode
        z, codes, latents, commitment_loss, codebook_loss = dac_model.encode(audio_tensor)
        print(f"Encoded codes shape: {codes.shape}")
        print(f"Codes dtype: {codes.dtype}")
        print(f"Codes range: {codes.min().item()} to {codes.max().item()}")
        
        # Decode back to audio
        reconstructed = dac_model.decode(z)
        print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Save reconstructed audio
    reconstructed_np = reconstructed.squeeze().cpu().numpy()
    sf.write("reconstructed_sine_wave.wav", reconstructed_np, sr)
    print(f"Saved reconstructed audio to: reconstructed_sine_wave.wav")
    
    # Save original for comparison
    original_np = waveform.squeeze().cpu().numpy()
    sf.write("original_sine_wave.wav", original_np, sr)
    print(f"Saved original audio to: original_sine_wave.wav")
    
    # Calculate metrics
    if len(reconstructed_np) != len(original_np):
        min_len = min(len(reconstructed_np), len(original_np))
        original_np = original_np[:min_len]
        reconstructed_np = reconstructed_np[:min_len]
        print(f"Trimmed to {min_len} samples for comparison")
    
    # Calculate reconstruction quality metrics
    mse = np.mean((original_np - reconstructed_np) ** 2)
    mae = np.mean(np.abs(original_np - reconstructed_np))
    max_error = np.max(np.abs(original_np - reconstructed_np))
    
    print(f"\nReconstruction Quality:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  Max Error: {max_error:.6f}")
    print(f"  Original range: [{original_np.min():.3f}, {original_np.max():.3f}]")
    print(f"  Reconstructed range: [{reconstructed_np.min():.3f}, {reconstructed_np.max():.3f}]")
    
    # Analyze codebook usage
    print(f"\nCodebook Analysis:")
    print(f"  Total codebooks: {codes.shape[1]}")
    print(f"  Time steps: {codes.shape[2]}")
    print(f"  Compression ratio: {waveform.shape[-1] / codes.shape[-1]:.1f}x")
    
    for i in range(codes.shape[1]):
        unique_codes = torch.unique(codes[0, i]).numel()
        most_common = torch.mode(codes[0, i]).values.item()
        print(f"  Codebook {i}: {unique_codes} unique codes, most common: {most_common}")
    
    # Check if reconstruction is reasonable for a sine wave
    if mse < 0.01:
        print(f"\n✅ DAC reconstruction looks good (MSE < 0.01)")
    elif mse < 0.1:
        print(f"\n⚠️  DAC reconstruction has some quality loss (MSE = {mse:.4f})")
    else:
        print(f"\n❌ DAC reconstruction quality is poor (MSE = {mse:.4f})")
    
    print(f"\nListening test:")
    print(f"  1. Play original_sine_wave.wav")
    print(f"  2. Play reconstructed_sine_wave.wav")
    print(f"  3. Compare if they sound the same")
    
    return codes, reconstructed


if __name__ == "__main__":
    test_dac_reconstruction()