#!/usr/bin/env python3
"""Test DAC encoding/decoding on sine wave with correct tensor format."""

import torch
import torchaudio
import soundfile as sf
import numpy as np
from pathlib import Path
import dac

from dia.config import DiaConfig


def test_dac_sine_wave():
    """Test DAC encoding/decoding quality on sine wave."""
    print("Loading DAC model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dac_model = dac.DAC.load(dac.utils.download()).to(device)
    
    print("Loading sine wave...")
    audio_path = Path("test_sine_wave_30s.wav")
    waveform, sr = torchaudio.load(audio_path)
    print(f"Original audio shape: {waveform.shape}, sample rate: {sr}")
    
    # Ensure proper format: (batch_size, channels, sequence_length)
    if waveform.dim() == 1:
        # (samples,) -> (1, 1, samples)
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.dim() == 2:
        if waveform.shape[0] > waveform.shape[1]:
            # Likely (samples, channels) -> transpose and add batch dim
            waveform = waveform.transpose(0, 1).unsqueeze(0)
        else:
            # Likely (channels, samples) -> add batch dim
            waveform = waveform.unsqueeze(0)
    
    print(f"Waveform shape for DAC: {waveform.shape}")
    print(f"Format: (batch_size={waveform.shape[0]}, channels={waveform.shape[1]}, samples={waveform.shape[2]})")
    waveform = waveform.to(device)
    
    print("Encoding with DAC...")
    with torch.no_grad():
        # Preprocess
        audio_tensor = dac_model.preprocess(waveform, sr).to(device)
        print(f"Preprocessed shape: {audio_tensor.shape}")
        
        # Encode
        z, codes, latents, commitment_loss, codebook_loss = dac_model.encode(audio_tensor)
        print(f"Codes shape: {codes.shape}")
        print(f"Codes dtype: {codes.dtype}")
        print(f"Codes range: {codes.min().item()} to {codes.max().item()}")
        
        # Decode
        reconstructed = dac_model.decode(z)
        print(f"Reconstructed shape: {reconstructed.shape}")
        
        # Postprocess
        reconstructed_audio = dac_model.postprocess(reconstructed, sr)
        print(f"Final reconstructed shape: {reconstructed_audio.shape}")
    
    # Save reconstructed audio
    reconstructed_np = reconstructed_audio.squeeze().cpu().numpy()
    sf.write("reconstructed_sine_wave.wav", reconstructed_np, sr)
    print("Saved reconstructed audio to: reconstructed_sine_wave.wav")
    
    # Calculate reconstruction error
    original_np = waveform.squeeze().cpu().numpy()
    if len(reconstructed_np) != len(original_np):
        min_len = min(len(reconstructed_np), len(original_np))
        original_np = original_np[:min_len]
        reconstructed_np = reconstructed_np[:min_len]
    
    mse = np.mean((original_np - reconstructed_np) ** 2)
    print(f"MSE between original and reconstructed: {mse:.6f}")
    
    # Check codebook stats
    print(f"\nCodebook analysis:")
    print(f"  Codes shape: {codes.shape}")  # Should be (batch, n_codebooks, time)
    print(f"  Number of codebooks: {codes.shape[1] if codes.dim() > 1 else 'N/A'}")
    print(f"  Time steps: {codes.shape[-1] if codes.dim() > 0 else 'N/A'}")
    print(f"  Unique codes per codebook:")
    for i in range(codes.shape[1]):
        unique_codes = torch.unique(codes[0, i]).numel()
        print(f"    Codebook {i}: {unique_codes} unique codes")
    
    return codes, reconstructed_audio


if __name__ == "__main__":
    test_dac_sine_wave()