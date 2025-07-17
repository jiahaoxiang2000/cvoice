"""Audio extraction from video files."""

import tempfile
from pathlib import Path

from moviepy.video.io.VideoFileClip import VideoFileClip

from ..utils.audio_utils import convert_audio_format
from ..utils.file_utils import ensure_dir, get_unique_filename
from .base import VideoProcessor


class AudioExtractor(VideoProcessor):
    """Extract audio from video files."""

    def __init__(
        self,
        output_format: str = "wav",
        sample_rate: int = 22050,
        channels: int = 1,
        output_dir: Path | None = None,
        **kwargs
    ) -> None:
        """Initialize audio extractor.
        
        Args:
            output_format: Output audio format (wav, mp3, etc.)
            sample_rate: Target sample rate
            channels: Number of audio channels
            output_dir: Directory to save extracted audio
            **kwargs: Additional configuration
        """
        super().__init__("AudioExtractor", **kwargs)
        self.output_format = output_format
        self.sample_rate = sample_rate
        self.channels = channels
        self.output_dir = output_dir or Path(tempfile.gettempdir()) / "cvoice_audio"

        ensure_dir(self.output_dir)

    def process(self, video_path: Path) -> Path:
        """Extract audio from video file.
        
        Args:
            video_path: Path to input video file
            
        Returns:
            Path to extracted audio file
            
        Raises:
            ValueError: If video file is invalid
            RuntimeError: If audio extraction fails
        """
        self.validate_input(video_path)

        self.logger.info(f"Extracting audio from: {video_path}")

        try:
            # Load video file
            with VideoFileClip(str(video_path)) as video:
                # Check if video has audio
                if video.audio is None:
                    raise ValueError(f"Video file has no audio track: {video_path}")

                # Generate output filename
                output_filename = f"{video_path.stem}_audio.{self.output_format}"
                output_path = self.output_dir / output_filename

                # Ensure unique filename
                output_path = get_unique_filename(
                    output_path.with_suffix(""),
                    self.output_format
                )

                # Extract audio
                audio_clip = video.audio

                # Set audio parameters
                audio_clip = audio_clip.set_fps(self.sample_rate)

                # Write audio file
                temp_path = output_path.with_suffix(".temp.wav")
                audio_clip.write_audiofile(
                    str(temp_path),
                    verbose=False,
                    logger=None
                )

                # Convert to desired format if needed
                if self.output_format.lower() != "wav":
                    final_path = convert_audio_format(
                        temp_path,
                        output_path,
                        self.output_format,
                        self.sample_rate,
                        self.channels
                    )
                    temp_path.unlink()  # Remove temporary file
                else:
                    final_path = temp_path.rename(output_path)

                self.logger.info(f"Audio extracted to: {final_path}")
                return final_path

        except Exception as e:
            self.logger.error(f"Failed to extract audio: {e}")
            raise RuntimeError(f"Audio extraction failed: {e}") from e

    def extract_audio_segment(
        self,
        video_path: Path,
        start_time: float,
        end_time: float
    ) -> Path:
        """Extract audio segment from video.
        
        Args:
            video_path: Path to input video file
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Path to extracted audio segment
        """
        self.validate_input(video_path)

        if start_time >= end_time:
            raise ValueError("Start time must be less than end time")

        self.logger.info(
            f"Extracting audio segment from {start_time}s to {end_time}s"
        )

        try:
            with VideoFileClip(str(video_path)) as video:
                if video.audio is None:
                    raise ValueError(f"Video file has no audio track: {video_path}")

                # Extract segment
                audio_segment = video.audio.subclip(start_time, end_time)

                # Generate output filename
                output_filename = (
                    f"{video_path.stem}_segment_{start_time:.1f}s-{end_time:.1f}s"
                    f".{self.output_format}"
                )
                output_path = self.output_dir / output_filename

                # Ensure unique filename
                output_path = get_unique_filename(
                    output_path.with_suffix(""),
                    self.output_format
                )

                # Set audio parameters and write file
                audio_segment = audio_segment.set_fps(self.sample_rate)

                temp_path = output_path.with_suffix(".temp.wav")
                audio_segment.write_audiofile(
                    str(temp_path),
                    verbose=False,
                    logger=None
                )

                # Convert to desired format if needed
                if self.output_format.lower() != "wav":
                    final_path = convert_audio_format(
                        temp_path,
                        output_path,
                        self.output_format,
                        self.sample_rate,
                        self.channels
                    )
                    temp_path.unlink()
                else:
                    final_path = temp_path.rename(output_path)

                self.logger.info(f"Audio segment extracted to: {final_path}")
                return final_path

        except Exception as e:
            self.logger.error(f"Failed to extract audio segment: {e}")
            raise RuntimeError(f"Audio segment extraction failed: {e}") from e

    def get_video_info(self, video_path: Path) -> dict:
        """Get video information.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        self.validate_input(video_path)

        try:
            with VideoFileClip(str(video_path)) as video:
                info = {
                    "duration": video.duration,
                    "fps": video.fps,
                    "size": video.size,
                    "has_audio": video.audio is not None
                }

                if video.audio is not None:
                    info["audio_fps"] = video.audio.fps
                    info["audio_duration"] = video.audio.duration

                return info

        except Exception as e:
            self.logger.error(f"Failed to get video info: {e}")
            raise RuntimeError(f"Video info extraction failed: {e}") from e
