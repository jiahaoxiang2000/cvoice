import pytest
from ..audio.merger import AudioMerger
import os
from unittest.mock import patch, MagicMock

def test_merge_audio_video(temp_dir, sample_video, sample_audio):
    output_path = os.path.join(temp_dir, "output.mp4")
    
    # Mock VideoFileClip and AudioFileClip since we can't test actual video processing
    with patch('audio.merger.VideoFileClip') as mock_video_clip, \
         patch('audio.merger.AudioFileClip') as mock_audio_clip:
        
        # Configure mocks
        mock_video = MagicMock()
        mock_audio = MagicMock()
        mock_video_clip.return_value = mock_video
        mock_audio_clip.return_value = mock_audio

        # Run the merger
        AudioMerger.merge_audio_video(sample_video, sample_audio, output_path)

        # Verify the expected calls
        mock_video_clip.assert_called_once_with(sample_video)
        mock_audio_clip.assert_called_once_with(sample_audio)
        mock_video.write_videofile.assert_called_once()
        mock_video.close.assert_called_once()
        mock_audio.close.assert_called_once()
