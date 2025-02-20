import pytest
import os
from ..audio.extractor import AudioExtractor
from ..utils.file_handler import FileHandler


def test_extract_audio(sample_video):
    before_path = sample_video.replace('.mp4', '.wav')
    if os.path.exists(before_path):
        os.remove(before_path)
    
    # Run the extractor
    output_path = AudioExtractor.extract_audio(sample_video)    
    
    # Verify output
    assert os.path.exists(output_path)
    assert output_path.endswith('.wav')
    assert os.path.getsize(output_path) > 0
    

def test_extract_audio_no_file():
    with pytest.raises(ValueError, match="Video file not found"):
        AudioExtractor.extract_audio("nonexistent.mp4")


def test_check_ffmpeg_version():
    # Verify FFMPEG is installed
    assert AudioExtractor.check_ffmpeg_version() is True


def test_validate_video_file_not_found():
    with pytest.raises(ValueError, match="Video file not found"):
        AudioExtractor.validate_video_file("nonexistent.mp4")
