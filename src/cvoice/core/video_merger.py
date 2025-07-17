"""Video and audio merging module."""

import tempfile
from pathlib import Path
from typing import Any

import moviepy.editor as mp
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip

from ..utils.audio_utils import get_audio_duration
from ..utils.file_utils import ensure_dir, get_unique_filename
from .base import VideoProcessor


class VideoMerger(VideoProcessor):
    """Merge synthesized audio with original video."""

    def __init__(
        self,
        output_format: str = "mp4",
        video_codec: str = "libx264",
        audio_codec: str = "aac",
        output_dir: Path | None = None,
        **kwargs
    ) -> None:
        """Initialize video merger.
        
        Args:
            output_format: Output video format
            video_codec: Video codec to use
            audio_codec: Audio codec to use
            output_dir: Directory to save merged videos
            **kwargs: Additional configuration
        """
        super().__init__("VideoMerger", **kwargs)
        self.output_format = output_format
        self.video_codec = video_codec
        self.audio_codec = audio_codec
        self.output_dir = output_dir or Path(tempfile.gettempdir()) / "cvoice_merged"

        ensure_dir(self.output_dir)

    def process(self, video_path: Path, audio_path: Path) -> Path:
        """Merge audio with video.
        
        Args:
            video_path: Path to video file
            audio_path: Path to audio file
            
        Returns:
            Path to merged video file
        """
        # Validate both inputs
        self.validate_input(video_path)
        self._validate_audio_input(audio_path)

        self.logger.info(f"Merging video: {video_path} with audio: {audio_path}")

        try:
            # Load video and audio
            with VideoFileClip(str(video_path)) as video:
                with AudioFileClip(str(audio_path)) as audio:
                    # Check duration compatibility
                    self._check_duration_compatibility(video, audio)

                    # Create output path
                    output_filename = f"{video_path.stem}_merged.{self.output_format}"
                    output_path = self.output_dir / output_filename
                    output_path = get_unique_filename(output_path.with_suffix(""), self.output_format)

                    # Replace video audio with new audio
                    final_video = video.set_audio(audio)

                    # Write merged video
                    final_video.write_videofile(
                        str(output_path),
                        codec=self.video_codec,
                        audio_codec=self.audio_codec,
                        verbose=False,
                        logger=None
                    )

                    self.logger.info(f"Video merged successfully: {output_path}")
                    return output_path

        except Exception as e:
            self.logger.error(f"Video merging failed: {e}")
            raise RuntimeError(f"Video merging failed: {e}") from e

    def merge_with_segments(
        self,
        video_path: Path,
        audio_segments: list[dict[str, Any]],
        output_path: Path | None = None
    ) -> Path:
        """Merge video with audio segments.
        
        Args:
            video_path: Path to video file
            audio_segments: List of audio segments with timing info
            output_path: Output video path
            
        Returns:
            Path to merged video file
        """
        self.validate_input(video_path)

        if not audio_segments:
            raise ValueError("No audio segments provided")

        self.logger.info(f"Merging video with {len(audio_segments)} audio segments")

        try:
            # Create output path if not provided
            if output_path is None:
                output_filename = f"{video_path.stem}_segment_merged.{self.output_format}"
                output_path = self.output_dir / output_filename
                output_path = get_unique_filename(output_path.with_suffix(""), self.output_format)

            # Load video
            with VideoFileClip(str(video_path)) as video:
                # Create composite audio from segments
                composite_audio = self._create_composite_audio(audio_segments, video.duration)

                # Replace video audio
                final_video = video.set_audio(composite_audio)

                # Write merged video
                final_video.write_videofile(
                    str(output_path),
                    codec=self.video_codec,
                    audio_codec=self.audio_codec,
                    verbose=False,
                    logger=None
                )

                self.logger.info(f"Segment-merged video created: {output_path}")
                return output_path

        except Exception as e:
            self.logger.error(f"Segment merging failed: {e}")
            raise RuntimeError(f"Segment merging failed: {e}") from e

    def replace_audio_segment(
        self,
        video_path: Path,
        audio_path: Path,
        start_time: float,
        end_time: float,
        output_path: Path | None = None
    ) -> Path:
        """Replace a specific audio segment in video.
        
        Args:
            video_path: Path to video file
            audio_path: Path to replacement audio
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Output video path
            
        Returns:
            Path to modified video file
        """
        self.validate_input(video_path)
        self._validate_audio_input(audio_path)

        if start_time >= end_time:
            raise ValueError("Start time must be less than end time")

        self.logger.info(f"Replacing audio segment {start_time:.2f}s-{end_time:.2f}s")

        try:
            # Create output path if not provided
            if output_path is None:
                output_filename = f"{video_path.stem}_segment_replaced.{self.output_format}"
                output_path = self.output_dir / output_filename
                output_path = get_unique_filename(output_path.with_suffix(""), self.output_format)

            with VideoFileClip(str(video_path)) as video:
                with AudioFileClip(str(audio_path)) as replacement_audio:
                    # Get original audio
                    original_audio = video.audio

                    # Create segments
                    before_segment = original_audio.subclip(0, start_time)
                    after_segment = original_audio.subclip(end_time, original_audio.duration)

                    # Adjust replacement audio duration
                    segment_duration = end_time - start_time
                    if replacement_audio.duration > segment_duration:
                        replacement_audio = replacement_audio.subclip(0, segment_duration)
                    elif replacement_audio.duration < segment_duration:
                        # Pad with silence if needed
                        silence_duration = segment_duration - replacement_audio.duration
                        silence = mp.AudioClip(lambda t: 0, duration=silence_duration)
                        replacement_audio = mp.concatenate_audioclips([replacement_audio, silence])

                    # Concatenate audio segments
                    new_audio = mp.concatenate_audioclips([
                        before_segment,
                        replacement_audio,
                        after_segment
                    ])

                    # Replace video audio
                    final_video = video.set_audio(new_audio)

                    # Write modified video
                    final_video.write_videofile(
                        str(output_path),
                        codec=self.video_codec,
                        audio_codec=self.audio_codec,
                        verbose=False,
                        logger=None
                    )

                    self.logger.info(f"Audio segment replaced: {output_path}")
                    return output_path

        except Exception as e:
            self.logger.error(f"Audio segment replacement failed: {e}")
            raise RuntimeError(f"Audio segment replacement failed: {e}") from e

    def adjust_audio_timing(
        self,
        video_path: Path,
        audio_path: Path,
        time_offset: float = 0.0,
        speed_factor: float = 1.0,
        output_path: Path | None = None
    ) -> Path:
        """Adjust audio timing when merging with video.
        
        Args:
            video_path: Path to video file
            audio_path: Path to audio file
            time_offset: Time offset in seconds
            speed_factor: Speed factor (1.0 = normal speed)
            output_path: Output video path
            
        Returns:
            Path to adjusted video file
        """
        self.validate_input(video_path)
        self._validate_audio_input(audio_path)

        self.logger.info(f"Adjusting audio timing: offset={time_offset:.2f}s, speed={speed_factor:.2f}x")

        try:
            # Create output path if not provided
            if output_path is None:
                output_filename = f"{video_path.stem}_timing_adjusted.{self.output_format}"
                output_path = self.output_dir / output_filename
                output_path = get_unique_filename(output_path.with_suffix(""), self.output_format)

            with VideoFileClip(str(video_path)) as video:
                with AudioFileClip(str(audio_path)) as audio:
                    # Adjust audio speed
                    if speed_factor != 1.0:
                        audio = audio.fx(mp.afx.speedx, speed_factor)

                    # Apply time offset
                    if time_offset != 0.0:
                        audio = audio.set_start(time_offset)

                    # Ensure audio fits video duration
                    if audio.duration > video.duration:
                        audio = audio.subclip(0, video.duration)
                    elif audio.duration < video.duration:
                        # Pad with silence
                        silence_duration = video.duration - audio.duration
                        silence = mp.AudioClip(lambda t: 0, duration=silence_duration)
                        audio = mp.concatenate_audioclips([audio, silence])

                    # Replace video audio
                    final_video = video.set_audio(audio)

                    # Write adjusted video
                    final_video.write_videofile(
                        str(output_path),
                        codec=self.video_codec,
                        audio_codec=self.audio_codec,
                        verbose=False,
                        logger=None
                    )

                    self.logger.info(f"Audio timing adjusted: {output_path}")
                    return output_path

        except Exception as e:
            self.logger.error(f"Audio timing adjustment failed: {e}")
            raise RuntimeError(f"Audio timing adjustment failed: {e}") from e

    def extract_original_audio(self, video_path: Path, output_path: Path | None = None) -> Path:
        """Extract original audio from video for backup.
        
        Args:
            video_path: Path to video file
            output_path: Output audio path
            
        Returns:
            Path to extracted audio file
        """
        self.validate_input(video_path)

        self.logger.info(f"Extracting original audio from: {video_path}")

        try:
            # Create output path if not provided
            if output_path is None:
                output_filename = f"{video_path.stem}_original_audio.wav"
                output_path = self.output_dir / output_filename
                output_path = get_unique_filename(output_path.with_suffix(""), "wav")

            with VideoFileClip(str(video_path)) as video:
                if video.audio is None:
                    raise ValueError("Video has no audio track")

                # Extract audio
                video.audio.write_audiofile(
                    str(output_path),
                    verbose=False,
                    logger=None
                )

                self.logger.info(f"Original audio extracted: {output_path}")
                return output_path

        except Exception as e:
            self.logger.error(f"Audio extraction failed: {e}")
            raise RuntimeError(f"Audio extraction failed: {e}") from e

    def _validate_audio_input(self, audio_path: Path) -> None:
        """Validate audio input file.
        
        Args:
            audio_path: Path to audio file
            
        Raises:
            ValueError: If audio file is invalid
        """
        if not audio_path.exists():
            raise ValueError(f"Audio file does not exist: {audio_path}")

        if not audio_path.is_file():
            raise ValueError(f"Audio path is not a file: {audio_path}")

        # Check for common audio extensions
        audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'}
        if audio_path.suffix.lower() not in audio_extensions:
            self.logger.warning(f"Unusual audio file extension: {audio_path.suffix}")

    def _check_duration_compatibility(self, video: VideoFileClip, audio: AudioFileClip) -> None:
        """Check if video and audio durations are compatible.
        
        Args:
            video: Video clip
            audio: Audio clip
        """
        video_duration = video.duration
        audio_duration = audio.duration

        duration_diff = abs(video_duration - audio_duration)

        if duration_diff > 1.0:  # More than 1 second difference
            self.logger.warning(
                f"Duration mismatch: video={video_duration:.2f}s, "
                f"audio={audio_duration:.2f}s, diff={duration_diff:.2f}s"
            )

    def _create_composite_audio(self, audio_segments: list[dict[str, Any]], video_duration: float) -> AudioFileClip:
        """Create composite audio from segments.
        
        Args:
            audio_segments: List of audio segments with timing info
            video_duration: Video duration for padding
            
        Returns:
            Composite audio clip
        """
        # Sort segments by start time
        sorted_segments = sorted(audio_segments, key=lambda x: x.get('start', 0))

        audio_clips = []
        current_time = 0.0

        for segment in sorted_segments:
            audio_path = Path(segment['audio_path'])
            start_time = segment.get('start', current_time)
            end_time = segment.get('end', start_time + get_audio_duration(audio_path))

            # Add silence if there's a gap
            if start_time > current_time:
                silence_duration = start_time - current_time
                silence = mp.AudioClip(lambda t: 0, duration=silence_duration)
                audio_clips.append(silence)

            # Add audio segment
            with AudioFileClip(str(audio_path)) as audio:
                segment_duration = end_time - start_time
                if audio.duration > segment_duration:
                    audio = audio.subclip(0, segment_duration)
                elif audio.duration < segment_duration:
                    # Pad with silence
                    silence_duration = segment_duration - audio.duration
                    silence = mp.AudioClip(lambda t: 0, duration=silence_duration)
                    audio = mp.concatenate_audioclips([audio, silence])

                audio_clips.append(audio)
                current_time = end_time

        # Add final silence if needed
        if current_time < video_duration:
            silence_duration = video_duration - current_time
            silence = mp.AudioClip(lambda t: 0, duration=silence_duration)
            audio_clips.append(silence)

        # Concatenate all audio clips
        composite_audio = mp.concatenate_audioclips(audio_clips)

        return composite_audio

    def get_video_info(self, video_path: Path) -> dict[str, Any]:
        """Get video information.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Video information dictionary
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
