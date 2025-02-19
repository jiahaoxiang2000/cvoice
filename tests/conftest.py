import pytest
import os
import tempfile
from pathlib import Path

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname

@pytest.fixture
def sample_video(temp_dir):
    video_path = Path(temp_dir) / "sample.mp4"
    # Create a dummy video file for testing
    with open(video_path, 'wb') as f:
        f.write(b'dummy video content')
    return str(video_path)

@pytest.fixture
def sample_audio(temp_dir):
    audio_path = Path(temp_dir) / "sample.wav"
    # Create a dummy audio file for testing
    with open(audio_path, 'wb') as f:
        f.write(b'dummy audio content')
    return str(audio_path)

@pytest.fixture
def sample_text():
    return "This is a test sentence."
