import speech_recognition as sr
import subprocess
from utils.logger import logger


class TextRecognizer:
    @staticmethod
    def audio_to_text(audio_file):
        recognizer = sr.Recognizer()

        # Use system FLAC instead of the bundled one
        flac_converter = subprocess.check_output(["which", "flac"]).decode().strip()
        recognizer.converter = flac_converter

        try:
            with sr.AudioFile(audio_file) as source:
                audio = recognizer.record(source)
                text = recognizer.recognize_google(audio)
                logger.info(f"Audio to text: {text}")
                return text
        except sr.UnknownValueError:
            logger.error("Could not understand audio")
            return "Could not understand audio"
        except sr.RequestError as e:
            logger.error(f"Could not request results: {str(e)}")
            return f"Could not request results; {str(e)}"
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            return f"Error processing audio: {str(e)}"
