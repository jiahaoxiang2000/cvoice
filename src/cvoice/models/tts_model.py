"""Text-to-speech model wrapper."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from TTS.api import TTS

from .base import BaseModel


class TTSModel(BaseModel):
    """Wrapper for TTS models with voice cloning capabilities."""

    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        device: str = "auto",
        use_gpu: bool = True,
        **kwargs
    ) -> None:
        """Initialize TTS model.
        
        Args:
            model_name: TTS model name
            device: Device to run on (cpu, cuda, auto)
            use_gpu: Whether to use GPU if available
            **kwargs: Additional model parameters
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.device = self._get_device(device, use_gpu)
        self.use_gpu = use_gpu

        # Initialize TTS API
        self.tts_api: TTS | None = None

    def _get_device(self, device: str, use_gpu: bool) -> str:
        """Get appropriate device for model.
        
        Args:
            device: Requested device
            use_gpu: Whether to use GPU
            
        Returns:
            Device string
        """
        if device == "auto":
            if use_gpu and torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device

    def load_model(self) -> None:
        """Load the TTS model."""
        if self.is_loaded:
            return

        try:
            self.tts_api = TTS(
                model_name=self.model_name,
                progress_bar=False,
                gpu=self.use_gpu
            )
            self._model = self.tts_api

        except Exception as e:
            raise RuntimeError(f"Failed to load TTS model: {e}") from e

    def unload_model(self) -> None:
        """Unload the model from memory."""
        if self._model is not None:
            del self._model
            del self.tts_api
            self._model = None
            self.tts_api = None

    def synthesize(
        self,
        text: str,
        output_path: Path,
        speaker_wav: Path | None = None,
        language: str = "en",
        **kwargs
    ) -> Path:
        """Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            output_path: Output audio file path
            speaker_wav: Reference speaker audio file for voice cloning
            language: Language code
            **kwargs: Additional synthesis parameters
            
        Returns:
            Path to synthesized audio file
        """
        if not self.is_loaded:
            self.load_model()

        try:
            if speaker_wav and speaker_wav.exists():
                # Voice cloning mode
                self.tts_api.tts_to_file(
                    text=text,
                    file_path=str(output_path),
                    speaker_wav=str(speaker_wav),
                    language=language,
                    **kwargs
                )
            else:
                # Standard TTS mode
                self.tts_api.tts_to_file(
                    text=text,
                    file_path=str(output_path),
                    language=language,
                    **kwargs
                )

            return output_path

        except Exception as e:
            raise RuntimeError(f"Speech synthesis failed: {e}") from e

    def clone_voice(
        self,
        text: str,
        reference_audio: Path,
        output_path: Path,
        language: str = "en",
        **kwargs
    ) -> Path:
        """Clone voice from reference audio.
        
        Args:
            text: Text to synthesize
            reference_audio: Reference audio file
            output_path: Output audio file path
            language: Language code
            **kwargs: Additional parameters
            
        Returns:
            Path to synthesized audio file
        """
        if not reference_audio.exists():
            raise ValueError(f"Reference audio file not found: {reference_audio}")

        return self.synthesize(
            text=text,
            output_path=output_path,
            speaker_wav=reference_audio,
            language=language,
            **kwargs
        )

    def batch_synthesize(
        self,
        texts: list[str],
        output_dir: Path,
        speaker_wav: Path | None = None,
        language: str = "en",
        **kwargs
    ) -> list[Path]:
        """Synthesize multiple texts.
        
        Args:
            texts: List of texts to synthesize
            output_dir: Output directory
            speaker_wav: Reference speaker audio file
            language: Language code
            **kwargs: Additional parameters
            
        Returns:
            List of output audio file paths
        """
        output_paths = []

        for i, text in enumerate(texts):
            output_path = output_dir / f"synthesis_{i:03d}.wav"

            try:
                result_path = self.synthesize(
                    text=text,
                    output_path=output_path,
                    speaker_wav=speaker_wav,
                    language=language,
                    **kwargs
                )
                output_paths.append(result_path)

            except Exception as e:
                print(f"Failed to synthesize text {i}: {e}")
                output_paths.append(None)

        return output_paths

    def get_supported_languages(self) -> list[str]:
        """Get list of supported languages.
        
        Returns:
            List of supported language codes
        """
        if not self.is_loaded:
            self.load_model()

        try:
            # Get languages from model
            if hasattr(self.tts_api, 'languages'):
                return self.tts_api.languages
            else:
                # Default languages for XTTS
                return ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko"]

        except Exception:
            return ["en"]  # Fallback to English

    def get_supported_speakers(self) -> list[str]:
        """Get list of supported speakers.
        
        Returns:
            List of supported speaker names
        """
        if not self.is_loaded:
            self.load_model()

        try:
            if hasattr(self.tts_api, 'speakers'):
                return self.tts_api.speakers
            else:
                return []

        except Exception:
            return []

    def analyze_reference_audio(self, audio_path: Path) -> dict[str, Any]:
        """Analyze reference audio for voice cloning.
        
        Args:
            audio_path: Path to reference audio file
            
        Returns:
            Analysis results
        """
        if not audio_path.exists():
            raise ValueError(f"Audio file not found: {audio_path}")

        try:
            # Load audio for analysis
            import librosa

            audio, sr = librosa.load(str(audio_path))

            # Basic audio analysis
            duration = len(audio) / sr
            rms_energy = float(np.sqrt(np.mean(audio**2)))

            # Spectral analysis
            spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
            zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(audio)))

            return {
                "duration": duration,
                "sample_rate": sr,
                "rms_energy": rms_energy,
                "spectral_centroid": spectral_centroid,
                "zero_crossing_rate": zero_crossing_rate,
                "suitable_for_cloning": duration >= 5.0,  # Minimum 5 seconds
                "quality_score": self._calculate_quality_score(rms_energy, spectral_centroid)
            }

        except Exception as e:
            raise RuntimeError(f"Audio analysis failed: {e}") from e

    def _calculate_quality_score(self, rms_energy: float, spectral_centroid: float) -> float:
        """Calculate quality score for reference audio.
        
        Args:
            rms_energy: RMS energy of audio
            spectral_centroid: Spectral centroid
            
        Returns:
            Quality score (0-1)
        """
        # Simple quality scoring based on audio characteristics
        energy_score = min(1.0, rms_energy * 10)  # Normalize energy
        spectral_score = min(1.0, spectral_centroid / 5000)  # Normalize spectral centroid

        return (energy_score + spectral_score) / 2

    def preprocess_text(self, text: str) -> str:
        """Preprocess text for better synthesis.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Basic text preprocessing
        import re

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Ensure proper punctuation
        if text and text[-1] not in '.!?':
            text += '.'

        return text

    def get_model_info(self) -> dict[str, Any]:
        """Get model information.
        
        Returns:
            Model information dictionary
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "use_gpu": self.use_gpu,
            "is_loaded": self.is_loaded,
            "supported_languages": self.get_supported_languages() if self.is_loaded else [],
            "supported_speakers": self.get_supported_speakers() if self.is_loaded else []
        }

    def estimate_synthesis_time(self, text: str) -> float:
        """Estimate synthesis time for given text.
        
        Args:
            text: Input text
            
        Returns:
            Estimated synthesis time in seconds
        """
        # Simple estimation based on text length
        # Approximate: 1 character = 0.05 seconds of speech
        return len(text) * 0.05

    def validate_reference_audio(self, audio_path: Path) -> bool:
        """Validate reference audio for voice cloning.
        
        Args:
            audio_path: Path to reference audio file
            
        Returns:
            True if audio is suitable for cloning
        """
        try:
            analysis = self.analyze_reference_audio(audio_path)
            return analysis["suitable_for_cloning"] and analysis["quality_score"] > 0.3

        except Exception:
            return False
