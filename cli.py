import argparse
from audio.extractor import AudioExtractor
from audio.synthesizer import AudioSynthesizer
from audio.merger import AudioMerger
from text.recognizer import TextRecognizer
from text.improver import TextImprover
from utils.logger import logger


def extract_audio(args):
    logger.info(f"Extracting audio from {args.input}")
    audio_file = AudioExtractor.extract_audio(args.input)
    logger.info(f"Audio extracted to {audio_file}")


def audio_to_text(args):
    logger.info(f"Converting audio to text: {args.input}")
    text = TextRecognizer.audio_to_text(args.input)
    print(f"Transcribed text: {text}")


def improve_text(args):
    logger.info("Improving text")
    improved = TextImprover.improve_text(args.input)
    print(f"Improved text: {improved}")


def text_to_audio(args):
    logger.info(f"Converting text to audio: {args.input}")
    audio_file = AudioSynthesizer.text_to_audio(args.input)
    logger.info(f"Audio generated at {audio_file}")


def merge_audio(args):
    logger.info(f"Merging audio {args.audio} with video {args.video}")
    AudioMerger.merge_audio_video(args.video, args.audio, args.output)
    logger.info(f"Merged file saved as {args.output}")


def full_pipeline(args):
    logger.info(f"Running full pipeline from {args.input} to {args.output}")
    audio_file = AudioExtractor.extract_audio(args.input)
    text = TextRecognizer.audio_to_text(audio_file)
    improved_text = TextImprover.improve_text(text)
    new_audio = AudioSynthesizer.text_to_audio(improved_text)
    AudioMerger.merge_audio_video(args.input, new_audio, args.output)
    logger.info(f"Pipeline complete. Output saved as {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="Voice conversion tool - Convert video speech from one voice to another"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Extract audio command
    extract_parser = subparsers.add_parser(
        "extract",
        aliases=["e"],
        help="Extract audio from video file",
        description="Extract the audio track from a video file and save it as WAV format",
    )
    extract_parser.add_argument(
        "-i", "--input", required=True, help="Path to input video file (mp4, avi, etc.)"
    )

    # Audio to text command
    audio2text_parser = subparsers.add_parser(
        "transcribe",
        aliases=["t"],
        help="Convert audio to text using speech recognition",
        description="Transcribe speech from an audio file to text using Google Speech Recognition",
    )
    audio2text_parser.add_argument(
        "-i", "--input", required=True, help="Path to input audio file (wav format)"
    )

    # Improve text command
    improve_parser = subparsers.add_parser(
        "improve",
        aliases=["i"],
        help="Improve text quality using AI",
        description="Enhance the text quality using BART language model",
    )
    improve_parser.add_argument("-i", "--input", required=True, help="Text to improve")

    # Text to audio command
    text2audio_parser = subparsers.add_parser(
        "synthesize",
        aliases=["s"],
        help="Convert text to speech",
        description="Generate speech audio from text using text-to-speech synthesis",
    )
    text2audio_parser.add_argument(
        "-i", "--input", required=True, help="Text to convert to speech"
    )

    # Merge audio command
    merge_parser = subparsers.add_parser(
        "merge",
        aliases=["m"],
        help="Merge audio with video",
        description="Combine a video file with a new audio track",
    )
    merge_parser.add_argument(
        "-v", "--video", required=True, help="Path to input video file"
    )
    merge_parser.add_argument(
        "-a", "--audio", required=True, help="Path to input audio file"
    )
    merge_parser.add_argument(
        "-o", "--output", required=True, help="Path for output video file"
    )

    # Full pipeline command
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        aliases=["p"],
        help="Run full conversion pipeline",
        description="Execute complete pipeline: extract audio → transcribe → improve → synthesize → merge",
    )
    pipeline_parser.add_argument(
        "-i", "--input", required=True, help="Path to input video file"
    )
    pipeline_parser.add_argument(
        "-o", "--output", required=True, help="Path for output video file"
    )

    args = parser.parse_args()

    if args.command == "extract":
        extract_audio(args)
    elif args.command == "transcribe":
        audio_to_text(args)
    elif args.command == "improve":
        improve_text(args)
    elif args.command == "synthesize":
        text_to_audio(args)
    elif args.command == "merge":
        merge_audio(args)
    elif args.command == "pipeline":
        full_pipeline(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
