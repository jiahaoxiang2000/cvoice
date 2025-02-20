import subprocess
import os
from moviepy import VideoFileClip
from ..utils.file_handler import FileHandler

class AudioExtractor:
    @staticmethod
    def check_ffmpeg_version():
        try:
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError("FFMPEG is not properly installed")
            return True
        except FileNotFoundError:
            raise RuntimeError("FFMPEG is not installed. Please install a recent version of FFMPEG")

    @staticmethod
    def validate_video_file(video_path):
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        
        # Check if file is not empty
        if os.path.getsize(video_path) == 0:
            raise ValueError(f"Video file is empty: {video_path}")
            
        # Check if file header is valid using ffprobe
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-select_streams', 'v:0', 
                 '-show_entries', 'stream=codec_name', '-of', 'default=noprint_wrappers=1:nokey=1', 
                 video_path],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise ValueError(f"Invalid video file format: {video_path}")
        except subprocess.SubprocessError as e:
            raise ValueError(f"Failed to validate video file: {str(e)}")

    @staticmethod
    def extract_audio(video_path):
        FileHandler.ensure_temp_dir()
        AudioExtractor.check_ffmpeg_version()
        AudioExtractor.validate_video_file(video_path)
        
        video = None
        audio = None
        try:
            video = VideoFileClip(video_path)
            if video.reader is None:
                raise OSError(f"Failed to read video file: {video_path}")
                
            audio = video.audio
            if audio is None:
                raise OSError(f"No audio stream found in video file: {video_path}")
                
            # on the video path, create a new file with the same name but with .wav extension
            audio_path = video_path.replace(".mp4", ".wav")
            audio.write_audiofile(audio_path)
            return audio_path 
            
        except Exception as e:
            error_msg = str(e)
            if "moov atom not found" in error_msg:
                raise RuntimeError(f"Video file is corrupted or invalid: {video_path}")
            elif "Invalid data found" in error_msg:
                raise RuntimeError(f"Invalid video format or corrupted file: {video_path}")
            else:
                raise RuntimeError(f"Error extracting audio: {error_msg}")
        finally:
            if video is not None:
                video.close()
            if audio is not None:
                audio.close()
