"""Speech-to-text conversion module."""

import json
import tempfile
from pathlib import Path
from typing import Any

from ..models.whisper_model import WhisperModel
from ..utils.audio_utils import get_audio_duration, load_audio, trim_silence
from ..utils.file_utils import ensure_dir, get_unique_filename
from .base import AudioProcessor


class SpeechToText(AudioProcessor):
    """Convert speech to text using various models."""

    def __init__(
        self,
        model_name: str = "base",
        device: str = "auto",
        language: str | None = None,
        use_faster_whisper: bool = True,
        output_dir: Path | None = None,
        **kwargs
    ) -> None:
        """Initialize speech-to-text converter.
        
        Args:
            model_name: Model name to use
            device: Device to run on
            language: Language code for transcription
            use_faster_whisper: Whether to use faster-whisper
            output_dir: Directory to save transcription results
            **kwargs: Additional configuration
        """
        super().__init__("SpeechToText", **kwargs)
        self.model_name = model_name
        self.device = device
        self.language = language
        self.use_faster_whisper = use_faster_whisper
        self.output_dir = output_dir or Path(tempfile.gettempdir()) / "cvoice_transcriptions"

        ensure_dir(self.output_dir)

        # Initialize model
        self.model = WhisperModel(
            model_name=model_name,
            device=device,
            use_faster_whisper=use_faster_whisper,
            **kwargs
        )

    def setup(self) -> None:
        """Set up the component."""
        super().setup()
        self.model.load_model()

    def teardown(self) -> None:
        """Tear down the component."""
        super().teardown()
        self.model.unload_model()

    def process(self, audio_path: Path) -> str:
        """Convert audio to text.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        self.validate_input(audio_path)

        self.logger.info(f"Transcribing audio: {audio_path}")

        try:
            # Get audio duration for logging
            duration = get_audio_duration(audio_path)
            self.logger.info(f"Audio duration: {duration:.2f} seconds")

            # Transcribe audio
            result = self.model.transcribe(
                audio_path,
                language=self.language,
                task="transcribe"
            )

            text = result["text"].strip()

            # Save transcription result
            self._save_transcription_result(audio_path, result)

            self.logger.info(f"Transcription completed: {len(text)} characters")
            return text

        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise RuntimeError(f"Speech-to-text conversion failed: {e}") from e

    def transcribe_with_timestamps(
        self,
        audio_path: Path,
        include_word_timestamps: bool = False
    ) -> list[dict[str, Any]]:
        """Transcribe audio with timestamps.
        
        Args:
            audio_path: Path to audio file
            include_word_timestamps: Whether to include word-level timestamps
            
        Returns:
            List of transcription segments with timestamps
        """
        self.validate_input(audio_path)

        self.logger.info(f"Transcribing with timestamps: {audio_path}")

        try:
            if include_word_timestamps:
                segments = self.model.transcribe_with_timestamps(
                    audio_path,
                    language=self.language
                )
            else:
                result = self.model.transcribe(
                    audio_path,
                    language=self.language,
                    task="transcribe"
                )
                segments = result.get("segments", [])

            # Save detailed transcription result
            self._save_detailed_transcription(audio_path, segments)

            self.logger.info(f"Timestamp transcription completed: {len(segments)} segments")
            return segments

        except Exception as e:
            self.logger.error(f"Timestamp transcription failed: {e}")
            raise RuntimeError(f"Timestamp transcription failed: {e}") from e

    def detect_language(self, audio_path: Path) -> dict[str, Any]:
        """Detect language of audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Language detection result
        """
        self.validate_input(audio_path)

        self.logger.info(f"Detecting language: {audio_path}")

        try:
            result = self.model.detect_language(audio_path)

            self.logger.info(
                f"Detected language: {result['language']} "
                f"(probability: {result['probability']:.2f})"
            )

            return result

        except Exception as e:
            self.logger.error(f"Language detection failed: {e}")
            raise RuntimeError(f"Language detection failed: {e}") from e

    def transcribe_segment(
        self,
        audio_path: Path,
        start_time: float,
        end_time: float
    ) -> str:
        """Transcribe a specific segment of audio.
        
        Args:
            audio_path: Path to audio file
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Transcribed text for the segment
        """
        self.validate_input(audio_path)

        if start_time >= end_time:
            raise ValueError("Start time must be less than end time")

        self.logger.info(
            f"Transcribing segment {start_time:.2f}s-{end_time:.2f}s: {audio_path}"
        )

        try:
            # Load audio and extract segment
            audio_data, sample_rate = load_audio(audio_path)

            # Convert time to sample indices
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)

            # Extract segment
            segment_audio = audio_data[start_sample:end_sample]

            # Save temporary segment file
            temp_path = self.output_dir / f"temp_segment_{start_time:.1f}s-{end_time:.1f}s.wav"

            import soundfile as sf
            sf.write(str(temp_path), segment_audio, sample_rate)

            try:
                # Transcribe segment
                result = self.model.transcribe(
                    temp_path,
                    language=self.language,
                    task="transcribe"
                )

                text = result["text"].strip()

                self.logger.info(f"Segment transcription completed: {len(text)} characters")
                return text

            finally:
                # Clean up temporary file
                if temp_path.exists():
                    temp_path.unlink()

        except Exception as e:
            self.logger.error(f"Segment transcription failed: {e}")
            raise RuntimeError(f"Segment transcription failed: {e}") from e

    def batch_transcribe(self, audio_paths: list[Path]) -> list[str]:
        """Transcribe multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            
        Returns:
            List of transcribed texts
        """
        results = []

        for i, audio_path in enumerate(audio_paths):
            self.logger.info(f"Processing file {i+1}/{len(audio_paths)}: {audio_path}")

            try:
                text = self.process(audio_path)
                results.append(text)
            except Exception as e:
                self.logger.error(f"Failed to transcribe {audio_path}: {e}")
                results.append("")  # Empty string for failed transcriptions

        return results

    def preprocess_audio(self, audio_path: Path) -> Path:
        """Preprocess audio for better transcription.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Path to preprocessed audio file
        """
        self.validate_input(audio_path)

        self.logger.info(f"Preprocessing audio: {audio_path}")

        try:
            # Load audio
            audio_data, sample_rate = load_audio(audio_path)

            # Trim silence
            trimmed_audio = trim_silence(audio_data, sample_rate)

            # Save preprocessed audio
            preprocessed_path = self.output_dir / f"preprocessed_{audio_path.name}"

            import soundfile as sf
            sf.write(str(preprocessed_path), trimmed_audio, sample_rate)

            self.logger.info(f"Preprocessed audio saved: {preprocessed_path}")
            return preprocessed_path

        except Exception as e:
            self.logger.error(f"Audio preprocessing failed: {e}")
            raise RuntimeError(f"Audio preprocessing failed: {e}") from e

    def _save_transcription_result(self, audio_path: Path, result: dict[str, Any]) -> None:
        """Save transcription result to file.
        
        Args:
            audio_path: Original audio file path
            result: Transcription result
        """
        try:
            output_filename = f"{audio_path.stem}_transcription.json"
            output_path = self.output_dir / output_filename

            # Ensure unique filename
            output_path = get_unique_filename(output_path.with_suffix(""), "json")

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            self.logger.debug(f"Transcription result saved: {output_path}")

        except Exception as e:
            self.logger.warning(f"Failed to save transcription result: {e}")

    def _save_detailed_transcription(self, audio_path: Path, segments: list[dict[str, Any]]) -> None:
        """Save detailed transcription with timestamps.
        
        Args:
            audio_path: Original audio file path
            segments: Transcription segments
        """
        try:
            output_filename = f"{audio_path.stem}_detailed_transcription.json"
            output_path = self.output_dir / output_filename

            # Ensure unique filename
            output_path = get_unique_filename(output_path.with_suffix(""), "json")

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(segments, f, indent=2, ensure_ascii=False)

            self.logger.debug(f"Detailed transcription saved: {output_path}")

        except Exception as e:
            self.logger.warning(f"Failed to save detailed transcription: {e}")

    def get_model_info(self) -> dict[str, Any]:
        """Get model information.
        
        Returns:
            Model information
        """
        return self.model.get_model_info()
