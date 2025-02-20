from TTS.api import TTS
from TTS.utils.synthesizer import Synthesizer
import torch
from ..utils.file_handler import FileHandler


class AudioSynthesizer:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize XTTS v2 model
        self.tts = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2"
        ).to(device)

    def text_to_audio(self, text, output_path=None):
        if output_path is None:
            FileHandler.ensure_temp_dir()
            output_path = FileHandler.get_temp_path("new_audio.wav")
        
        try:
            # Synthesize audio from text
            self.tts.tts_to_file(text, output_path)
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"Text-to-speech conversion failed: {str(e)}")
