import os
import sys
from moviepy.editor import VideoFileClip, AudioFileClip
import speech_recognition as sr
import pyttsx3
from transformers import pipeline

class CVoice:
    def __init__(self, input_video, output_video):
        self.input_video = input_video
        self.output_video = output_video
        self.temp_dir = "temp"
        os.makedirs(self.temp_dir, exist_ok=True)

    def extract_audio(self):
        """Extract audio from video"""
        video = VideoFileClip(self.input_video)
        audio = video.audio
        temp_audio = os.path.join(self.temp_dir, "temp_audio.wav")
        audio.write_audiofile(temp_audio)
        return temp_audio

    def audio_to_text(self, audio_file):
        """Convert audio to text using speech recognition"""
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
        return text

    def improve_text(self, text):
        """Improve text quality using transformer model"""
        corrector = pipeline("text2text-generation", model="facebook/bart-large-cnn")
        improved_text = corrector(text, max_length=130)[0]['generated_text']
        return improved_text

    def text_to_audio(self, text):
        """Convert text to audio"""
        engine = pyttsx3.init()
        temp_audio = os.path.join(self.temp_dir, "new_audio.wav")
        engine.save_to_file(text, temp_audio)
        engine.runAndWait()
        return temp_audio

    def merge_audio_video(self, new_audio):
        """Merge new audio with original video"""
        video = VideoFileClip(self.input_video)
        audio = AudioFileClip(new_audio)
        final_video = video.set_audio(audio)
        final_video.write_videofile(self.output_video)

    def process(self):
        """Run the entire voice conversion pipeline"""
        print("Extracting audio from video...")
        audio_file = self.extract_audio()

        print("Converting audio to text...")
        text = self.audio_to_text(audio_file)

        print("Improving text quality...")
        improved_text = self.improve_text(text)

        print("Converting text to new audio...")
        new_audio = self.text_to_audio(improved_text)

        print("Merging new audio with video...")
        self.merge_audio_video(new_audio)

        print("Done! Output saved as:", self.output_video)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python cvoice.py input_video.mp4 output_video.mp4")
        sys.exit(1)

    converter = CVoice(sys.argv[1], sys.argv[2])
    converter.process()
