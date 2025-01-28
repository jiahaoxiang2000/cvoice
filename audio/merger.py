from moviepy import VideoFileClip, AudioFileClip


class AudioMerger:
    @staticmethod
    def merge_audio_video(video_path, audio_path, output_path):
        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)
        final_video = video.set_audio(audio)
        final_video.write_videofile(output_path)
