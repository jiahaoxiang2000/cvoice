import whisper
import json
import os
from datetime import datetime
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

    @staticmethod
    def _save_to_file(result, output_path, format="txt"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "text": result["text"],
                        "segments": result["segments"],
                        "timestamp": datetime.now().isoformat(),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        elif format == "srt":
            with open(output_path, "w", encoding="utf-8") as f:
                for _, seg in enumerate(result["segments"], 1):
                    start = datetime.utcfromtimestamp(seg["start"]).strftime(
                        "%H:%M:%S,%f"
                    )[:-3]
                    end = datetime.utcfromtimestamp(seg["end"]).strftime("%H:%M:%S,%f")[
                        :-3
                    ]
                    f.write(f"{start} --> {end}\n")
                    f.write(f"{seg['text'].strip()}\n\n")

        else:  # txt format
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result["text"])

    @classmethod
    def audio_to_text(cls, audio_file, output_format="srt", output_dir="data"):
        try:
            model = cls.get_model()
            result = model.transcribe(audio_file)

            # Create output filename
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                output_dir, f"{base_name}_{timestamp}.{output_format}"
            )

            # Save to file
            cls._save_to_file(result, output_path, output_format)

            logger.info(f"Transcription saved to: {output_path}")
            return result["text"]

        except Exception as e:
            error_msg = f"Error processing audio: {str(e)}"
            logger.error(error_msg)
            return error_msg
