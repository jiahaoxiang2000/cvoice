"""Pytest configuration and fixtures."""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_audio_file(temp_dir: Path) -> Path:
    """Create a sample audio file for testing."""
    # This would normally create a real audio file
    audio_file = temp_dir / "sample.wav"
    audio_file.write_text("fake audio content")
    return audio_file


@pytest.fixture
def sample_video_file(temp_dir: Path) -> Path:
    """Create a sample video file for testing."""
    # This would normally create a real video file
    video_file = temp_dir / "sample.mp4"
    video_file.write_text("fake video content")
    return video_file