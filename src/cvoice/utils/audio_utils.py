"""Audio utility functions."""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment


def get_audio_duration(audio_path: str | Path) -> float:
    """Get audio duration in seconds.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Duration in seconds
    """
    audio = AudioSegment.from_file(str(audio_path))
    return len(audio) / 1000.0  # Convert milliseconds to seconds


def normalize_audio(audio_data: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """Normalize audio to target dB level.
    
    Args:
        audio_data: Audio data as numpy array
        target_db: Target dB level
        
    Returns:
        Normalized audio data
    """
    # Convert to AudioSegment for normalization
    audio_segment = AudioSegment(
        audio_data.tobytes(),
        frame_rate=22050,  # Default sample rate
        sample_width=audio_data.dtype.itemsize,
        channels=1
    )

    # Normalize to target dB
    normalized = audio_segment.normalize().apply_gain(target_db)

    # Convert back to numpy array
    return np.array(normalized.get_array_of_samples(), dtype=np.float32)


def convert_audio_format(
    input_path: str | Path,
    output_path: str | Path,
    output_format: str = "wav",
    sample_rate: int = 22050,
    channels: int = 1
) -> Path:
    """Convert audio file format.
    
    Args:
        input_path: Input audio file path
        output_path: Output audio file path
        output_format: Output format (wav, mp3, etc.)
        sample_rate: Target sample rate
        channels: Number of channels (1 for mono, 2 for stereo)
        
    Returns:
        Path to converted file
    """
    # Load audio file
    audio = AudioSegment.from_file(str(input_path))

    # Convert to target format
    if channels == 1:
        audio = audio.set_channels(1)
    elif channels == 2:
        audio = audio.set_channels(2)

    audio = audio.set_frame_rate(sample_rate)

    # Export to target format
    output_path = Path(output_path)
    audio.export(str(output_path), format=output_format)

    return output_path


def load_audio(audio_path: str | Path, sample_rate: int = 22050) -> tuple[np.ndarray, int]:
    """Load audio file using librosa.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    audio_data, sr = librosa.load(str(audio_path), sr=sample_rate)
    return audio_data, sr


def save_audio(
    audio_data: np.ndarray,
    output_path: str | Path,
    sample_rate: int = 22050
) -> Path:
    """Save audio data to file.
    
    Args:
        audio_data: Audio data as numpy array
        output_path: Output file path
        sample_rate: Sample rate
        
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    sf.write(str(output_path), audio_data, sample_rate)
    return output_path


def extract_audio_features(
    audio_data: np.ndarray,
    sample_rate: int = 22050,
    n_mfcc: int = 13
) -> dict:
    """Extract audio features for analysis.
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate
        n_mfcc: Number of MFCC coefficients
        
    Returns:
        Dictionary of extracted features
    """
    features = {}

    # MFCC features
    features['mfcc'] = librosa.feature.mfcc(
        y=audio_data, sr=sample_rate, n_mfcc=n_mfcc
    )

    # Spectral features
    features['spectral_centroid'] = librosa.feature.spectral_centroid(
        y=audio_data, sr=sample_rate
    )
    features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
        y=audio_data, sr=sample_rate
    )
    features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(audio_data)

    # Tempo and beat
    features['tempo'], features['beats'] = librosa.beat.beat_track(
        y=audio_data, sr=sample_rate
    )

    return features


def trim_silence(
    audio_data: np.ndarray,
    sample_rate: int = 22050,
    top_db: int = 20
) -> np.ndarray:
    """Trim silence from beginning and end of audio.
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate
        top_db: dB below peak to consider as silence
        
    Returns:
        Trimmed audio data
    """
    trimmed, _ = librosa.effects.trim(audio_data, top_db=top_db)
    return trimmed
