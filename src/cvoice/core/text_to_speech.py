"""Text-to-speech with voice cloning capabilities."""

import json
import tempfile
from pathlib import Path
from typing import Any

from ..models.tts_model import TTSModel
from ..utils.audio_utils import get_audio_duration
from ..utils.file_utils import ensure_dir, get_unique_filename
from .base import TextProcessor


class TextToSpeech(TextProcessor):
    """Convert text to speech with voice cloning."""

    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        device: str = "auto",
        language: str = "en",
        reference_audio: Path | None = None,
        output_dir: Path | None = None,
        **kwargs
    ) -> None:
        """Initialize text-to-speech converter.
        
        Args:
            model_name: TTS model name
            device: Device to run on
            language: Language code
            reference_audio: Reference audio for voice cloning
            output_dir: Directory to save synthesized audio
            **kwargs: Additional configuration
        """
        super().__init__("TextToSpeech", **kwargs)
        self.model_name = model_name
        self.device = device
        self.language = language
        self.reference_audio = reference_audio
        self.output_dir = output_dir or Path(tempfile.gettempdir()) / "cvoice_synthesis"

        ensure_dir(self.output_dir)

        # Initialize TTS model
        self.model = TTSModel(
            model_name=model_name,
            device=device,
            **kwargs
        )

        # Validate reference audio if provided
        if self.reference_audio and not self._validate_reference_audio():
            raise ValueError(f"Invalid reference audio: {self.reference_audio}")

    def setup(self) -> None:
        """Set up the component."""
        super().setup()
        self.model.load_model()

    def teardown(self) -> None:
        """Tear down the component."""
        super().teardown()
        self.model.unload_model()

    def process(self, input_text: str) -> Path:
        """Convert text to speech.
        
        Args:
            input_text: Text to convert
            
        Returns:
            Path to synthesized audio file
        """
        self.validate_input(input_text)

        self.logger.info(f"Synthesizing speech: {len(input_text)} characters")

        try:
            # Preprocess text
            preprocessed_text = self.model.preprocess_text(input_text)

            # Generate output filename
            output_filename = "synthesized_speech.wav"
            output_path = self.output_dir / output_filename

            # Ensure unique filename
            output_path = get_unique_filename(output_path.with_suffix(""), "wav")

            # Synthesize speech
            result_path = self.model.synthesize(
                text=preprocessed_text,
                output_path=output_path,
                speaker_wav=self.reference_audio,
                language=self.language
            )

            # Log synthesis info
            duration = get_audio_duration(result_path)
            self.logger.info(f"Speech synthesized: {duration:.2f} seconds")

            # Save synthesis metadata
            self._save_synthesis_metadata(input_text, result_path)

            return result_path

        except Exception as e:
            self.logger.error(f"Speech synthesis failed: {e}")
            raise RuntimeError(f"Text-to-speech conversion failed: {e}") from e

    def synthesize_with_reference(
        self,
        text: str,
        reference_audio: Path,
        output_path: Path | None = None
    ) -> Path:
        """Synthesize speech with specific reference audio.
        
        Args:
            text: Text to synthesize
            reference_audio: Reference audio file
            output_path: Output audio file path
            
        Returns:
            Path to synthesized audio file
        """
        self.validate_input(text)

        if not reference_audio.exists():
            raise ValueError(f"Reference audio not found: {reference_audio}")

        # Validate reference audio
        if not self.model.validate_reference_audio(reference_audio):
            self.logger.warning(f"Reference audio may not be suitable: {reference_audio}")

        self.logger.info(f"Synthesizing with reference: {reference_audio}")

        try:
            # Generate output path if not provided
            if output_path is None:
                output_filename = f"cloned_speech_{reference_audio.stem}.wav"
                output_path = self.output_dir / output_filename
                output_path = get_unique_filename(output_path.with_suffix(""), "wav")

            # Synthesize speech
            result_path = self.model.clone_voice(
                text=text,
                reference_audio=reference_audio,
                output_path=output_path,
                language=self.language
            )

            # Log synthesis info
            duration = get_audio_duration(result_path)
            self.logger.info(f"Voice cloned speech synthesized: {duration:.2f} seconds")

            return result_path

        except Exception as e:
            self.logger.error(f"Voice cloning failed: {e}")
            raise RuntimeError(f"Voice cloning failed: {e}") from e

    def batch_synthesize(
        self,
        texts: list[str],
        reference_audio: Path | None = None,
        output_prefix: str = "batch_synthesis"
    ) -> list[Path]:
        """Synthesize multiple texts.
        
        Args:
            texts: List of texts to synthesize
            reference_audio: Reference audio for voice cloning
            output_prefix: Prefix for output files
            
        Returns:
            List of synthesized audio file paths
        """
        self.logger.info(f"Batch synthesizing {len(texts)} texts")

        output_paths = []
        speaker_wav = reference_audio or self.reference_audio

        for i, text in enumerate(texts):
            try:
                self.logger.info(f"Processing text {i+1}/{len(texts)}")

                # Generate output path
                output_filename = f"{output_prefix}_{i:03d}.wav"
                output_path = self.output_dir / output_filename
                output_path = get_unique_filename(output_path.with_suffix(""), "wav")

                # Synthesize
                result_path = self.model.synthesize(
                    text=text,
                    output_path=output_path,
                    speaker_wav=speaker_wav,
                    language=self.language
                )

                output_paths.append(result_path)

            except Exception as e:
                self.logger.error(f"Failed to synthesize text {i+1}: {e}")
                output_paths.append(None)

        successful_syntheses = [p for p in output_paths if p is not None]
        self.logger.info(f"Batch synthesis completed: {len(successful_syntheses)}/{len(texts)} successful")

        return output_paths

    def synthesize_segments(
        self,
        text_segments: list[dict[str, Any]],
        reference_audio: Path | None = None
    ) -> list[Path]:
        """Synthesize text segments with timestamps.
        
        Args:
            text_segments: List of text segments with timing info
            reference_audio: Reference audio for voice cloning
            
        Returns:
            List of synthesized audio file paths
        """
        self.logger.info(f"Synthesizing {len(text_segments)} segments")

        output_paths = []
        speaker_wav = reference_audio or self.reference_audio

        for i, segment in enumerate(text_segments):
            try:
                text = segment.get("text", "")
                start_time = segment.get("start", 0)
                end_time = segment.get("end", 0)

                if not text.strip():
                    continue

                # Generate output path
                output_filename = f"segment_{i:03d}_{start_time:.1f}s-{end_time:.1f}s.wav"
                output_path = self.output_dir / output_filename
                output_path = get_unique_filename(output_path.with_suffix(""), "wav")

                # Synthesize segment
                result_path = self.model.synthesize(
                    text=text,
                    output_path=output_path,
                    speaker_wav=speaker_wav,
                    language=self.language
                )

                output_paths.append(result_path)

            except Exception as e:
                self.logger.error(f"Failed to synthesize segment {i}: {e}")
                output_paths.append(None)

        return output_paths

    def set_reference_audio(self, reference_audio: Path) -> None:
        """Set reference audio for voice cloning.
        
        Args:
            reference_audio: Path to reference audio file
        """
        if not reference_audio.exists():
            raise ValueError(f"Reference audio not found: {reference_audio}")

        if not self.model.validate_reference_audio(reference_audio):
            self.logger.warning(f"Reference audio may not be suitable: {reference_audio}")

        self.reference_audio = reference_audio
        self.logger.info(f"Reference audio set: {reference_audio}")

    def analyze_reference_audio(self, audio_path: Path | None = None) -> dict[str, Any]:
        """Analyze reference audio for voice cloning.
        
        Args:
            audio_path: Path to audio file (uses default if not provided)
            
        Returns:
            Analysis results
        """
        target_audio = audio_path or self.reference_audio

        if not target_audio or not target_audio.exists():
            raise ValueError("No valid reference audio provided")

        return self.model.analyze_reference_audio(target_audio)

    def get_supported_languages(self) -> list[str]:
        """Get supported languages.
        
        Returns:
            List of supported language codes
        """
        return self.model.get_supported_languages()

    def get_supported_speakers(self) -> list[str]:
        """Get supported speakers.
        
        Returns:
            List of supported speaker names
        """
        return self.model.get_supported_speakers()

    def estimate_synthesis_time(self, text: str) -> float:
        """Estimate synthesis time for text.
        
        Args:
            text: Input text
            
        Returns:
            Estimated synthesis time in seconds
        """
        return self.model.estimate_synthesis_time(text)

    def _validate_reference_audio(self) -> bool:
        """Validate reference audio file.
        
        Returns:
            True if reference audio is valid
        """
        if not self.reference_audio:
            return True  # No reference audio is fine

        if not self.reference_audio.exists():
            return False

        try:
            # Check audio duration
            duration = get_audio_duration(self.reference_audio)
            if duration < 3.0:  # Minimum 3 seconds
                self.logger.warning(f"Reference audio too short: {duration:.2f}s")
                return False

            # Additional validation can be added here
            return True

        except Exception as e:
            self.logger.error(f"Reference audio validation failed: {e}")
            return False

    def _save_synthesis_metadata(self, text: str, audio_path: Path) -> None:
        """Save synthesis metadata.
        
        Args:
            text: Original text
            audio_path: Path to synthesized audio
        """
        try:
            metadata = {
                "text": text,
                "audio_path": str(audio_path),
                "model_name": self.model_name,
                "language": self.language,
                "reference_audio": str(self.reference_audio) if self.reference_audio else None,
                "duration": get_audio_duration(audio_path),
                "synthesis_time": self.estimate_synthesis_time(text)
            }

            metadata_filename = f"{audio_path.stem}_metadata.json"
            metadata_path = self.output_dir / metadata_filename

            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            self.logger.debug(f"Synthesis metadata saved: {metadata_path}")

        except Exception as e:
            self.logger.warning(f"Failed to save synthesis metadata: {e}")

    def get_model_info(self) -> dict[str, Any]:
        """Get model information.
        
        Returns:
            Model information
        """
        return self.model.get_model_info()
