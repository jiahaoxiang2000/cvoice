import pytest
from ..audio.extractor import AudioExtractor
import os
from unittest.mock import patch, MagicMock

def test_extract_audio(sample_video):
    with patch('audio.extractor.VideoFileClip') as mock_video_clip:
        # Configure mocks
        mock_video = MagicMock()
        mock_audio = MagicMock()
        mock_video.audio = mock_audio
        mock_video_clip.return_value = mock_video

        # Run the extractor
        output_path = AudioExtractor.extract_audio(sample_video)

        # Verify the calls
        mock_video_clip.assert_called_once_with(sample_video)
        mock_audio.write_audiofile.assert_called_once()
        mock_video.close.assert_called_once()
        mock_audio.close.assert_called_once()
        assert output_path.endswith('.wav')

def test_extract_audio_no_video():
    with pytest.raises(Exception):
        AudioExtractor.extract_audio("nonexistent.mp4")
