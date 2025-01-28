import sys
from audio.extractor import AudioExtractor
from audio.synthesizer import AudioSynthesizer
from audio.merger import AudioMerger
from text.recognizer import TextRecognizer
from text.improver import TextImprover


def main():
    if len(sys.argv) != 3:
        print("Usage: python -m cvoice.cli input_video.mp4 output_video.mp4")
        sys.exit(1)

    input_video = sys.argv[1]
    output_video = sys.argv[2]

    print("Extracting audio from video...")
    audio_file = AudioExtractor.extract_audio(input_video)

    print("Converting audio to text...")
    text = TextRecognizer.audio_to_text(audio_file)

    print("Improving text quality...")
    improved_text = TextImprover.improve_text(text)

    print("Converting text to new audio...")
    new_audio = AudioSynthesizer.text_to_audio(improved_text)

    print("Merging new audio with video...")
    AudioMerger.merge_audio_video(input_video, new_audio, output_video)

    print("Done! Output saved as:", output_video)


if __name__ == "__main__":
    main()
