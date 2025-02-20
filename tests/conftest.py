import pytest
import os
import tempfile
from pathlib import Path

@pytest.fixture
def temp_dir():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    return str(data_dir)

@pytest.fixture
def sample_video(temp_dir):
    video_path = Path(temp_dir) / "sample.mp4"
    return str(video_path)

@pytest.fixture
def sample_audio(temp_dir):
    audio_path = Path(temp_dir) / "sample.wav"
    return str(audio_path)

@pytest.fixture
def sample_text():
    return "This is a test sentence."
