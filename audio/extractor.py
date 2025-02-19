from moviepy import VideoFileClip
from ..utils.file_handler import FileHandler


class AudioExtractor:
    @staticmethod
    def extract_audio(video_path):
        FileHandler.ensure_temp_dir()
        video = VideoFileClip(video_path)
        audio = video.audio
        temp_audio = FileHandler.get_temp_path("extracted_audio.wav")
        audio.write_audiofile(temp_audio)
        video.close()
        audio.close()
        return temp_audio
