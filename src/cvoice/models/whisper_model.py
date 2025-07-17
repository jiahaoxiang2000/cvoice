"""Whisper model wrapper."""

from pathlib import Path
from typing import Any

import whisper
from faster_whisper import WhisperModel as FasterWhisperModel

from .base import BaseModel


class WhisperModel(BaseModel):
    """Wrapper for OpenAI Whisper model."""

    def __init__(
        self,
        model_name: str = "base",
        device: str = "auto",
        use_faster_whisper: bool = True,
        **kwargs
    ) -> None:
        """Initialize Whisper model.
        
        Args:
            model_name: Whisper model name (tiny, base, small, medium, large)
            device: Device to run on (cpu, cuda, auto)
            use_faster_whisper: Whether to use faster-whisper implementation
            **kwargs: Additional model parameters
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.device = device
        self.use_faster_whisper = use_faster_whisper

        # Validate model name
        valid_models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        if model_name not in valid_models:
            raise ValueError(f"Invalid model name: {model_name}. Must be one of {valid_models}")

    def load_model(self) -> None:
        """Load the Whisper model."""
        if self.is_loaded:
            return

        try:
            if self.use_faster_whisper:
                self._model = FasterWhisperModel(
                    self.model_name,
                    device=self.device,
                    **self.config
                )
            else:
                self._model = whisper.load_model(
                    self.model_name,
                    device=self.device
                )
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {e}") from e

    def unload_model(self) -> None:
        """Unload the model from memory."""
        if self._model is not None:
            del self._model
            self._model = None

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        task: str = "transcribe",
        **kwargs
    ) -> dict[str, Any]:
        """Transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'es', 'fr')
            task: Task type ('transcribe' or 'translate')
            **kwargs: Additional transcription parameters
            
        Returns:
            Transcription result dictionary
        """
        if not self.is_loaded:
            self.load_model()

        try:
            if self.use_faster_whisper:
                segments, info = self._model.transcribe(
                    str(audio_path),
                    language=language,
                    task=task,
                    **kwargs
                )

                # Convert segments to list for easier handling
                segments_list = []
                for segment in segments:
                    segments_list.append({
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text
                    })

                return {
                    "text": " ".join([seg["text"] for seg in segments_list]),
                    "segments": segments_list,
                    "language": info.language,
                    "language_probability": info.language_probability
                }
            else:
                result = self._model.transcribe(
                    str(audio_path),
                    language=language,
                    task=task,
                    **kwargs
                )
                return result

        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}") from e

    def transcribe_with_timestamps(
        self,
        audio_path: Path,
        language: str | None = None,
        **kwargs
    ) -> list[dict[str, Any]]:
        """Transcribe audio with word-level timestamps.
        
        Args:
            audio_path: Path to audio file
            language: Language code
            **kwargs: Additional transcription parameters
            
        Returns:
            List of segments with timestamps
        """
        if not self.is_loaded:
            self.load_model()

        try:
            if self.use_faster_whisper:
                segments, _ = self._model.transcribe(
                    str(audio_path),
                    language=language,
                    word_timestamps=True,
                    **kwargs
                )

                result = []
                for segment in segments:
                    segment_dict = {
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text,
                        "words": []
                    }

                    if hasattr(segment, 'words') and segment.words:
                        for word in segment.words:
                            segment_dict["words"].append({
                                "word": word.word,
                                "start": word.start,
                                "end": word.end,
                                "probability": word.probability
                            })

                    result.append(segment_dict)

                return result
            else:
                result = self._model.transcribe(
                    str(audio_path),
                    language=language,
                    word_timestamps=True,
                    **kwargs
                )
                return result.get("segments", [])

        except Exception as e:
            raise RuntimeError(f"Timestamp transcription failed: {e}") from e

    def detect_language(self, audio_path: Path) -> dict[str, Any]:
        """Detect language of audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Language detection result
        """
        if not self.is_loaded:
            self.load_model()

        try:
            if self.use_faster_whisper:
                segments, info = self._model.transcribe(
                    str(audio_path),
                    language=None,
                    task="transcribe",
                    without_timestamps=True,
                    max_initial_timestamp=30.0
                )

                # Get first segment to determine language
                first_segment = next(segments, None)

                return {
                    "language": info.language,
                    "probability": info.language_probability,
                    "all_language_probs": info.all_language_probs if hasattr(info, 'all_language_probs') else None
                }
            else:
                # Load a small portion for language detection
                audio = whisper.load_audio(str(audio_path))
                audio = whisper.pad_or_trim(audio)

                # Make log-Mel spectrogram
                mel = whisper.log_mel_spectrogram(audio).to(self._model.device)

                # Detect language
                _, probs = self._model.detect_language(mel)

                return {
                    "language": max(probs, key=probs.get),
                    "probability": max(probs.values()),
                    "all_language_probs": probs
                }

        except Exception as e:
            raise RuntimeError(f"Language detection failed: {e}") from e

    def get_model_info(self) -> dict[str, Any]:
        """Get model information.
        
        Returns:
            Model information dictionary
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "use_faster_whisper": self.use_faster_whisper,
            "is_loaded": self.is_loaded
        }
