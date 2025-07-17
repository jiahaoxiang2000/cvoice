"""Main voice cloning pipeline."""

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..utils.file_utils import ensure_dir, get_unique_filename
from ..utils.logging_utils import get_logger
from .audio_extractor import AudioExtractor
from .base import Pipeline
from .speech_to_text import SpeechToText
from .text_improver import TextImprover
from .text_to_speech import TextToSpeech
from .video_merger import VideoMerger


@dataclass
class PipelineConfig:
    """Configuration for voice cloning pipeline."""
    # Audio extraction settings
    audio_format: str = "wav"
    audio_sample_rate: int = 22050
    audio_channels: int = 1

    # Speech-to-text settings
    stt_model: str = "base"
    stt_language: str | None = None
    stt_device: str = "auto"

    # Text improvement settings
    text_improvement_enabled: bool = True
    text_api_provider: str = "openai"
    text_api_key: str | None = None
    text_model: str = "gpt-3.5-turbo"

    # Text-to-speech settings
    tts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    tts_language: str = "en"
    tts_device: str = "auto"

    # Video merging settings
    video_format: str = "mp4"
    video_codec: str = "libx264"
    audio_codec: str = "aac"

    # Output settings
    output_dir: Path | None = None
    keep_intermediate_files: bool = False


@dataclass
class PipelineResult:
    """Result of voice cloning pipeline."""
    input_video: Path
    output_video: Path
    reference_audio: Path | None
    extracted_audio: Path
    transcribed_text: str
    improved_text: str
    synthesized_audio: Path
    processing_time: float
    success: bool
    error_message: str | None = None
    intermediate_files: list[Path] = None


class VoiceClonePipeline:
    """Main voice cloning pipeline orchestrator."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """Initialize voice cloning pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self.logger = get_logger("VoiceClonePipeline")

        # Set up output directory
        self.output_dir = self.config.output_dir or Path(tempfile.gettempdir()) / "cvoice_pipeline"
        ensure_dir(self.output_dir)

        # Initialize components
        self._initialize_components()

        # Create pipeline
        self.pipeline = Pipeline("VoiceClonePipeline")

    def _initialize_components(self) -> None:
        """Initialize pipeline components."""
        # Audio extractor
        self.audio_extractor = AudioExtractor(
            output_format=self.config.audio_format,
            sample_rate=self.config.audio_sample_rate,
            channels=self.config.audio_channels,
            output_dir=self.output_dir / "extracted_audio"
        )

        # Speech-to-text
        self.speech_to_text = SpeechToText(
            model_name=self.config.stt_model,
            device=self.config.stt_device,
            language=self.config.stt_language,
            output_dir=self.output_dir / "transcriptions"
        )

        # Text improver (optional)
        if self.config.text_improvement_enabled:
            self.text_improver = TextImprover(
                api_provider=self.config.text_api_provider,
                api_key=self.config.text_api_key,
                model_name=self.config.text_model,
                output_dir=self.output_dir / "improved_text"
            )
        else:
            self.text_improver = None

        # Text-to-speech
        self.text_to_speech = TextToSpeech(
            model_name=self.config.tts_model,
            device=self.config.tts_device,
            language=self.config.tts_language,
            output_dir=self.output_dir / "synthesized_audio"
        )

        # Video merger
        self.video_merger = VideoMerger(
            output_format=self.config.video_format,
            video_codec=self.config.video_codec,
            audio_codec=self.config.audio_codec,
            output_dir=self.output_dir / "merged_videos"
        )

    def process_video(
        self,
        input_video: Path,
        reference_audio: Path | None = None,
        output_path: Path | None = None
    ) -> PipelineResult:
        """Process video through the complete voice cloning pipeline.
        
        Args:
            input_video: Path to input video file
            reference_audio: Path to reference audio for voice cloning
            output_path: Path for output video
            
        Returns:
            Pipeline processing result
        """
        import time
        start_time = time.time()

        self.logger.info(f"Starting voice cloning pipeline for: {input_video}")

        intermediate_files = []

        try:
            # Step 1: Extract audio from video
            self.logger.info("Step 1: Extracting audio from video")
            extracted_audio = self.audio_extractor.process(input_video)
            intermediate_files.append(extracted_audio)

            # Step 2: Convert speech to text
            self.logger.info("Step 2: Converting speech to text")
            transcribed_text = self.speech_to_text.process(extracted_audio)

            # Step 3: Improve text (optional)
            if self.text_improver:
                self.logger.info("Step 3: Improving transcribed text")
                improved_text = self.text_improver.process(transcribed_text)
            else:
                improved_text = transcribed_text

            # Step 4: Convert text to speech with voice cloning
            self.logger.info("Step 4: Converting text to speech")
            if reference_audio:
                self.text_to_speech.set_reference_audio(reference_audio)
            synthesized_audio = self.text_to_speech.process(improved_text)
            intermediate_files.append(synthesized_audio)

            # Step 5: Merge synthesized audio with original video
            self.logger.info("Step 5: Merging audio with video")
            if output_path is None:
                output_filename = f"{input_video.stem}_voice_cloned.{self.config.video_format}"
                output_path = self.output_dir / output_filename
                output_path = get_unique_filename(output_path.with_suffix(""), self.config.video_format)

            output_video = self.video_merger.process(input_video, synthesized_audio)

            # Move to final output path if different
            if output_video != output_path:
                output_video.rename(output_path)
                output_video = output_path

            processing_time = time.time() - start_time
            self.logger.info(f"Pipeline completed successfully in {processing_time:.2f} seconds")

            # Clean up intermediate files if requested
            if not self.config.keep_intermediate_files:
                self._cleanup_intermediate_files(intermediate_files)
                intermediate_files = []

            return PipelineResult(
                input_video=input_video,
                output_video=output_video,
                reference_audio=reference_audio,
                extracted_audio=extracted_audio,
                transcribed_text=transcribed_text,
                improved_text=improved_text,
                synthesized_audio=synthesized_audio,
                processing_time=processing_time,
                success=True,
                intermediate_files=intermediate_files
            )

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Pipeline failed after {processing_time:.2f} seconds: {e}")

            # Clean up intermediate files on error
            self._cleanup_intermediate_files(intermediate_files)

            return PipelineResult(
                input_video=input_video,
                output_video=Path(),
                reference_audio=reference_audio,
                extracted_audio=Path(),
                transcribed_text="",
                improved_text="",
                synthesized_audio=Path(),
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )

    def process_with_segments(
        self,
        input_video: Path,
        reference_audio: Path | None = None,
        segment_timestamps: list[dict[str, Any]] | None = None,
        output_path: Path | None = None
    ) -> PipelineResult:
        """Process video with custom segment timestamps.
        
        Args:
            input_video: Path to input video file
            reference_audio: Path to reference audio for voice cloning
            segment_timestamps: List of segments with start/end times
            output_path: Path for output video
            
        Returns:
            Pipeline processing result
        """
        import time
        start_time = time.time()

        self.logger.info(f"Starting segmented voice cloning pipeline for: {input_video}")

        intermediate_files = []

        try:
            # Step 1: Extract audio from video
            extracted_audio = self.audio_extractor.process(input_video)
            intermediate_files.append(extracted_audio)

            # Step 2: Transcribe with timestamps
            if segment_timestamps:
                # Use provided segments
                transcription_segments = segment_timestamps
            else:
                # Get segments from speech-to-text
                transcription_segments = self.speech_to_text.transcribe_with_timestamps(extracted_audio)

            # Step 3: Improve text for each segment
            if self.text_improver:
                improved_segments = []
                for segment in transcription_segments:
                    improved_text = self.text_improver.process(segment.get('text', ''))
                    improved_segment = segment.copy()
                    improved_segment['text'] = improved_text
                    improved_segments.append(improved_segment)
            else:
                improved_segments = transcription_segments

            # Step 4: Synthesize audio for each segment
            if reference_audio:
                self.text_to_speech.set_reference_audio(reference_audio)

            synthesized_segments = []
            for i, segment in enumerate(improved_segments):
                segment_text = segment.get('text', '')
                if segment_text.strip():
                    segment_audio = self.text_to_speech.process(segment_text)
                    synthesized_segments.append({
                        'audio_path': str(segment_audio),
                        'start': segment.get('start', 0),
                        'end': segment.get('end', 0)
                    })
                    intermediate_files.append(segment_audio)

            # Step 5: Merge segments with video
            if output_path is None:
                output_filename = f"{input_video.stem}_voice_cloned_segments.{self.config.video_format}"
                output_path = self.output_dir / output_filename
                output_path = get_unique_filename(output_path.with_suffix(""), self.config.video_format)

            output_video = self.video_merger.merge_with_segments(
                input_video,
                synthesized_segments,
                output_path
            )

            processing_time = time.time() - start_time
            self.logger.info(f"Segmented pipeline completed in {processing_time:.2f} seconds")

            # Clean up intermediate files if requested
            if not self.config.keep_intermediate_files:
                self._cleanup_intermediate_files(intermediate_files)
                intermediate_files = []

            # Combine all text for result
            full_transcribed_text = " ".join([seg.get('text', '') for seg in transcription_segments])
            full_improved_text = " ".join([seg.get('text', '') for seg in improved_segments])

            return PipelineResult(
                input_video=input_video,
                output_video=output_video,
                reference_audio=reference_audio,
                extracted_audio=extracted_audio,
                transcribed_text=full_transcribed_text,
                improved_text=full_improved_text,
                synthesized_audio=Path(),  # Multiple files
                processing_time=processing_time,
                success=True,
                intermediate_files=intermediate_files
            )

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Segmented pipeline failed: {e}")

            # Clean up intermediate files on error
            self._cleanup_intermediate_files(intermediate_files)

            return PipelineResult(
                input_video=input_video,
                output_video=Path(),
                reference_audio=reference_audio,
                extracted_audio=Path(),
                transcribed_text="",
                improved_text="",
                synthesized_audio=Path(),
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )

    def batch_process(
        self,
        input_videos: list[Path],
        reference_audio: Path | None = None,
        output_dir: Path | None = None
    ) -> list[PipelineResult]:
        """Process multiple videos.
        
        Args:
            input_videos: List of input video paths
            reference_audio: Path to reference audio for voice cloning
            output_dir: Directory for output videos
            
        Returns:
            List of pipeline results
        """
        self.logger.info(f"Starting batch processing for {len(input_videos)} videos")

        results = []

        for i, video_path in enumerate(input_videos):
            self.logger.info(f"Processing video {i+1}/{len(input_videos)}: {video_path}")

            try:
                # Generate output path
                if output_dir:
                    output_path = output_dir / f"{video_path.stem}_voice_cloned.{self.config.video_format}"
                else:
                    output_path = None

                result = self.process_video(video_path, reference_audio, output_path)
                results.append(result)

            except Exception as e:
                self.logger.error(f"Failed to process video {i+1}: {e}")
                results.append(PipelineResult(
                    input_video=video_path,
                    output_video=Path(),
                    reference_audio=reference_audio,
                    extracted_audio=Path(),
                    transcribed_text="",
                    improved_text="",
                    synthesized_audio=Path(),
                    processing_time=0.0,
                    success=False,
                    error_message=str(e)
                ))

        successful_results = [r for r in results if r.success]
        self.logger.info(f"Batch processing completed: {len(successful_results)}/{len(input_videos)} successful")

        return results

    def save_result(self, result: PipelineResult, output_path: Path) -> None:
        """Save pipeline result to JSON file.
        
        Args:
            result: Pipeline result
            output_path: Path to save result
        """
        try:
            result_dict = asdict(result)

            # Convert Path objects to strings for JSON serialization
            for key, value in result_dict.items():
                if isinstance(value, Path):
                    result_dict[key] = str(value)
                elif isinstance(value, list) and value and isinstance(value[0], Path):
                    result_dict[key] = [str(p) for p in value]

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Pipeline result saved: {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to save pipeline result: {e}")

    def _cleanup_intermediate_files(self, files: list[Path]) -> None:
        """Clean up intermediate files.
        
        Args:
            files: List of file paths to clean up
        """
        for file_path in files:
            try:
                if file_path.exists():
                    file_path.unlink()
                    self.logger.debug(f"Cleaned up intermediate file: {file_path}")
            except Exception as e:
                self.logger.warning(f"Failed to clean up {file_path}: {e}")

    def get_pipeline_info(self) -> dict[str, Any]:
        """Get pipeline configuration and component information.
        
        Returns:
            Pipeline information dictionary
        """
        return {
            "config": asdict(self.config),
            "components": {
                "audio_extractor": "AudioExtractor",
                "speech_to_text": self.speech_to_text.get_model_info(),
                "text_improver": "TextImprover" if self.text_improver else None,
                "text_to_speech": self.text_to_speech.get_model_info(),
                "video_merger": "VideoMerger"
            },
            "output_dir": str(self.output_dir)
        }
