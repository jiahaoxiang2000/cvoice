import whisper
from utils.logger import logger

# Add version check and logging
logger.info(f"Using Whisper version: {whisper.__version__}")


class TextRecognizer:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            # Load the model only once and cache it
            # Options: tiny, base, small, medium, large
            cls._model = whisper.load_model("medium")
        return cls._model

    @classmethod
    def audio_to_text(cls, audio_file):
        try:
            model = cls.get_model()
            result = model.transcribe(audio_file)
            text = result["text"]
            logger.info(f"Audio to text: {text}")
            return text
        except Exception as e:
            error_msg = f"Error processing audio: {str(e)}"
            logger.error(error_msg)
            return error_msg
