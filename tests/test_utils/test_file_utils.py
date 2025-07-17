"""Tests for file utilities."""

import pytest
from pathlib import Path
import tempfile
import shutil

from cvoice.utils.file_utils import (
    ensure_dir,
    get_file_extension,
    validate_file_path,
    get_unique_filename
)


class TestFileUtils:
    """Test file utility functions."""
    
    def test_ensure_dir_creates_directory(self, temp_dir):
        """Test directory creation."""
        test_dir = temp_dir / "test_dir"
        result = ensure_dir(test_dir)
        
        assert result == test_dir
        assert test_dir.exists()
        assert test_dir.is_dir()
        
    def test_ensure_dir_existing_directory(self, temp_dir):
        """Test with existing directory."""
        result = ensure_dir(temp_dir)
        
        assert result == temp_dir
        assert temp_dir.exists()
        
    def test_get_file_extension(self):
        """Test file extension extraction."""
        assert get_file_extension("test.txt") == ".txt"
        assert get_file_extension("test.MP4") == ".mp4"
        assert get_file_extension(Path("test.wav")) == ".wav"
        assert get_file_extension("test") == ""
        
    def test_validate_file_path_existing_file(self, temp_dir):
        """Test file path validation with existing file."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        
        result = validate_file_path(test_file)
        assert result == test_file
        
    def test_validate_file_path_nonexistent_file(self, temp_dir):
        """Test file path validation with non-existent file."""
        test_file = temp_dir / "nonexistent.txt"
        
        with pytest.raises(ValueError, match="File does not exist"):
            validate_file_path(test_file)
            
    def test_validate_file_path_directory(self, temp_dir):
        """Test file path validation with directory."""
        with pytest.raises(ValueError, match="Path is not a file"):
            validate_file_path(temp_dir)
            
    def test_validate_file_path_not_required(self, temp_dir):
        """Test file path validation when file doesn't need to exist."""
        test_file = temp_dir / "nonexistent.txt"
        
        result = validate_file_path(test_file, must_exist=False)
        assert result == test_file
        
    def test_get_unique_filename_new_file(self, temp_dir):
        """Test unique filename generation for new file."""
        base_path = temp_dir / "test"
        result = get_unique_filename(base_path, "txt")
        
        assert result == temp_dir / "test.txt"
        
    def test_get_unique_filename_existing_file(self, temp_dir):
        """Test unique filename generation for existing file."""
        base_path = temp_dir / "test"
        
        # Create existing file
        existing_file = temp_dir / "test.txt"
        existing_file.write_text("existing")
        
        result = get_unique_filename(base_path, "txt")
        
        assert result == temp_dir / "test_1.txt"
        
    def test_get_unique_filename_multiple_existing(self, temp_dir):
        """Test unique filename generation with multiple existing files."""
        base_path = temp_dir / "test"
        
        # Create multiple existing files
        (temp_dir / "test.txt").write_text("test")
        (temp_dir / "test_1.txt").write_text("test")
        (temp_dir / "test_2.txt").write_text("test")
        
        result = get_unique_filename(base_path, "txt")
        
        assert result == temp_dir / "test_3.txt"