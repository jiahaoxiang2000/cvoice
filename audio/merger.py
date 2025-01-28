from moviepy import VideoFileClip, AudioFileClip


class AudioMerger:
    @staticmethod
    def merge_audio_video(video_path, audio_path, output_path):
        try:
            video = VideoFileClip(video_path)
            audio = AudioFileClip(audio_path)

            # Set the audio using the correct method
            video.audio = audio

            # Write the output file with the original video codec
            video.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                temp_audiofile="temp-audio.m4a",
                remove_temp=True,
            )

        finally:
            # Clean up resources
            video.close()
            audio.close()
