"""Simple sine wave dataset for testing music generation."""

import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from torch.utils.data import Dataset
import dac

from .config import DiaConfig


class SineWaveDataset(Dataset):
    """Dataset that repeatedly returns the same sine wave for overfitting tests."""
    
    def __init__(self, audio_path: Path, text: str, config: DiaConfig, dac_model: dac.DAC):
        """
        Args:
            audio_path: Path to the sine wave audio file
            text: Text description (e.g., "sine wave")
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
        """Encode the sine wave audio using DAC."""
        # Load audio
        audio_data, sr = sf.read(self.audio_path)
        
        # Convert to tensor with proper shape (1, 1, samples) for mono
        if audio_data.ndim == 1:
            waveform = torch.from_numpy(audio_data).float().unsqueeze(0).unsqueeze(0)
        else:
            # Handle stereo by taking first channel
            waveform = torch.from_numpy(audio_data[:, 0]).float().unsqueeze(0).unsqueeze(0)
        
        print(f"Loading audio: {self.audio_path}")
        print(f"Audio shape: {waveform.shape}, sample rate: {sr}")
        
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
        
    def __len__(self) -> int:
        # Single sample for true overfitting
        return 1
        
    def __getitem__(self, idx: int):
        """Return the same sine wave sample every time."""
        # Return: (text, encoded_audio, waveform_placeholder)
        # waveform_placeholder is not used in training but expected by collate_fn
        waveform_placeholder = torch.zeros(1, 1000)  # Dummy waveform
        
        return self.text, self.encoded_audio, waveform_placeholder


def create_sine_dataset(
    frequency: float = 440.0,
    duration: float = 30.0,
    sample_rate: int = 44100,
    text: str = "sine wave",
    output_path: Path = Path("sine_wave_dataset.wav")
) -> Path:
    """Create a sine wave audio file for the dataset.
    
    Args:
        frequency: Frequency in Hz
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        text: Text description
        output_path: Where to save the audio file
        
    Returns:
        Path to the created audio file
    """
    # Generate sine wave
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sine_wave = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Save as mono audio
    sf.write(output_path, sine_wave.astype(np.float32), sample_rate)
    print(f"Created sine wave dataset: {output_path}")
    print(f"  Frequency: {frequency} Hz")
    print(f"  Duration: {duration} seconds")
    print(f"  Text: '{text}'")
    
    return output_path


if __name__ == "__main__":
    # Test the dataset creation
    from .config import DiaConfig
    
    # Create sine wave file
    audio_path = create_sine_dataset()
    
    # Load config and DAC model
    config = DiaConfig.load("dia/music_config.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dac_model = dac.DAC.load(dac.utils.download()).to(device)
    
    # Create dataset
    dataset = SineWaveDataset(audio_path, "sine wave", config, dac_model)
    
    # Test getting an item
    text, encoded, waveform = dataset[0]
    print(f"Dataset item - Text: '{text}', Encoded shape: {encoded.shape}")