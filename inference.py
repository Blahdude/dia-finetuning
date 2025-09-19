#!/usr/bin/env python3
"""Universal inference script for trained Dia models."""

import argparse
import torch
import soundfile as sf
import numpy as np
from pathlib import Path

from dia.config import DiaConfig
from dia.model import Dia


def seconds_to_tokens(T_seconds: float, r: float = 86.1328125) -> int:
    """Convert seconds to tokens using calibrated rate."""
    return int(round(r * T_seconds))


def find_checkpoint(checkpoint_dir: str, epoch: int = None) -> Path:
    """Find the best available checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    
    if epoch:
        # Look for specific epoch
        specific_path = checkpoint_dir / f"ckpt_epoch{epoch}.pth"
        if specific_path.exists():
            return specific_path
        print(f"WARNING: Epoch {epoch} not found, looking for alternatives...")
    
    # Find all checkpoints
    checkpoints = list(checkpoint_dir.glob("ckpt_epoch*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    # Sort by epoch number (extract number from filename)
    def extract_epoch(path):
        try:
            return int(path.stem.split('epoch')[1])
        except:
            return 0
    
    checkpoints.sort(key=extract_epoch, reverse=True)
    
    print(f"Available checkpoints:")
    for ckpt in checkpoints[:5]:  # Show top 5
        epoch_num = extract_epoch(ckpt)
        print(f"  Epoch {epoch_num}: {ckpt}")
    
    return checkpoints[0]  # Return highest epoch


def universal_inference(
    text_prompt: str,
    checkpoint_path: str,
    duration_seconds: float = 6.0,
    config_path: str = "dia/music_config.json",
    temperature: float = 1.5,
    top_p: float = 0.8,
    cfg_scale: float = 1.0,
    output_path: str = None,
    epoch: int = None
):
    """Universal inference function for any trained model.
    
    Args:
        text_prompt: Text description for generation
        checkpoint_path: Path to checkpoint directory or specific .pth file
        duration_seconds: How many seconds of audio to generate
        config_path: Path to model config
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        cfg_scale: Classifier-free guidance scale
        output_path: Where to save generated audio
        epoch: Specific epoch to load (optional)
    """
    print(f"Universal Dia Inference")
    print(f"Text prompt: '{text_prompt}'")
    print(f"Target duration: {duration_seconds} seconds")
    
    # Load config
    config_path = Path(config_path)
    config = DiaConfig.load(config_path)
    print(f"Loaded config from: {config_path}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Find checkpoint
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_file() and checkpoint_path.suffix == '.pth':
        # Direct path to checkpoint file
        final_checkpoint = checkpoint_path
    else:
        # Directory - find best checkpoint
        final_checkpoint = find_checkpoint(checkpoint_path, epoch)
    
    print(f"Loading checkpoint: {final_checkpoint}")
    
    # Create Dia model with trained weights
    dia = Dia.from_local(
        config_path=str(config_path),
        checkpoint_path=str(final_checkpoint),
        device=device
    )
    
    print(f"Generating audio for text: '{text_prompt}'")
    
    # Calculate target duration and tokens
    target_tokens = seconds_to_tokens(duration_seconds)
    max_tokens = min(target_tokens, config.data.audio_length)
    
    if target_tokens > config.data.audio_length:
        actual_max_duration = config.data.audio_length / 86.1328125
        print(f"WARNING: Requested {duration_seconds}s exceeds max length")
        print(f"Generating maximum length: {actual_max_duration:.1f}s instead")
    
    print(f"Target tokens: {target_tokens}")
    print(f"Using max_tokens: {max_tokens}")
    print(f"Generation settings: temp={temperature}, top_p={top_p}, cfg_scale={cfg_scale}")
    
    # Generate audio
    generated_audio = dia.generate(
        text=text_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        cfg_scale=cfg_scale
    )
    
    # Debug: Check generated audio properties
    print(f"Generated audio shape: {generated_audio.shape}")
    print(f"Generated audio range: [{generated_audio.min():.3f}, {generated_audio.max():.3f}]")
    print(f"Generated audio dtype: {generated_audio.dtype}")
    print(f"Contains NaN: {np.isnan(generated_audio).any()}")
    print(f"Contains Inf: {np.isinf(generated_audio).any()}")
    
    # Determine output filename
    if output_path is None:
        safe_prompt = "".join(c if c.isalnum() or c in ' -_' else '' for c in text_prompt)
        safe_prompt = safe_prompt.replace(' ', '_').lower()
        output_path = f"generated_{safe_prompt}.wav"
    
    # Save generated audio
    sf.write(output_path, generated_audio, 44100)
    
    actual_duration = len(generated_audio) / 44100
    print(f"\nResults:")
    print(f"  Saved to: {output_path}")
    print(f"  Target duration: {duration_seconds}s")
    print(f"  Actual duration: {actual_duration:.2f}s")
    print(f"  Duration accuracy: {(actual_duration/duration_seconds)*100:.1f}%")
    print(f"  Audio range: [{generated_audio.min():.3f}, {generated_audio.max():.3f}]")
    print(f"  Audio shape: {generated_audio.shape}")
    
    return generated_audio, output_path


def main():
    parser = argparse.ArgumentParser(description="Generate audio from trained Dia model")
    parser.add_argument("text", type=str, help="Text prompt for generation")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint directory or .pth file")
    parser.add_argument("--duration", type=float, default=6.0, help="Duration in seconds (default: 6.0)")
    parser.add_argument("--config", type=str, default="dia/music_config.json", help="Path to model config")
    parser.add_argument("--temperature", type=float, default=1.5, help="Sampling temperature (default: 1.5)")
    parser.add_argument("--top_p", type=float, default=0.8, help="Top-p sampling (default: 0.8)")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="CFG scale (default: 1.0)")
    parser.add_argument("--output", type=str, default=None, help="Output filename (auto-generated if not specified)")
    parser.add_argument("--epoch", type=int, default=None, help="Specific epoch to load")
    
    args = parser.parse_args()
    
    try:
        generated_audio, output_path = universal_inference(
            text_prompt=args.text,
            checkpoint_path=args.checkpoint,
            duration_seconds=args.duration,
            config_path=args.config,
            temperature=args.temperature,
            top_p=args.top_p,
            cfg_scale=args.cfg_scale,
            output_path=args.output,
            epoch=args.epoch
        )
        
        print(f"\n✅ Success! Generated audio saved to: {output_path}")
        print(f"🎵 Play with: ffplay {output_path}")
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())