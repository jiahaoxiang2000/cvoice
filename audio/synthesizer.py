import pyttsx3
from utils.file_handler import FileHandler


class AudioSynthesizer:
    @staticmethod
    def text_to_audio(text):
        FileHandler.ensure_temp_dir()
        engine = pyttsx3.init()
        temp_audio = FileHandler.get_temp_path("new_audio.wav")
        engine.save_to_file(text, temp_audio)
        engine.runAndWait()
        return temp_audio
