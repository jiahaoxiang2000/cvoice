import subprocess
from utils.file_handler import FileHandler


class AudioSynthesizer:
    @staticmethod
    def text_to_audio(text):
        FileHandler.ensure_temp_dir()
        temp_audio = FileHandler.get_temp_path("new_audio.wav")
        aiff_audio = FileHandler.get_temp_path("temp.aiff")

        try:
            # First create AIFF file (macOS say command native format)
            subprocess.run(
                ["say", "-v", "Alex", "-o", aiff_audio, text],
                check=True,
                capture_output=True,
                text=True,
            )

            # Convert AIFF to WAV using afconvert
            subprocess.run(
                ["afconvert", "-f", "WAVE", "-d", "LEI16", aiff_audio, temp_audio],
                check=True,
                capture_output=True,
                text=True,
            )

            # Clean up temporary AIFF file
            subprocess.run(["rm", aiff_audio], check=True)
            
            return temp_audio

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            raise RuntimeError(f"Text-to-speech conversion failed: {error_msg}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during text-to-speech: {str(e)}")
