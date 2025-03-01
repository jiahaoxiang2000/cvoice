import whisper
import json
import os
from datetime import datetime, timezone
from ..utils.logger import logger


class TextRecognizer:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            # Load the model only once and cache it
            # Options: tiny, base, small, medium, large
            cls._model = whisper.load_model("turbo")
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
                    start = datetime.fromtimestamp(seg["start"], tz=timezone.utc).strftime(
                        "%H:%M:%S"
                    )
                    end = datetime.fromtimestamp(seg["end"], tz=timezone.utc).strftime("%H:%M:%S")
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
            output_path = os.path.join(output_dir, f"{base_name}.{output_format}")

            # Save to file
            cls._save_to_file(result, output_path, output_format)

            logger.info(f"Transcription saved to: {output_path}")
            return result["text"]

        except Exception as e:
            error_msg = f"Error processing audio: {str(e)}"
            logger.error(error_msg)
            return error_msg
