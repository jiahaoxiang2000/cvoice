import pytest
import os
import json
from ..text.recognizer import TextRecognizer

@pytest.fixture
def sample_audio_path(sample_audio):
    return sample_audio

@pytest.fixture
def output_dir(sample_audio):
    return os.path.dirname(sample_audio)
     

def test_model_initialization():
    model = TextRecognizer.get_model()
    assert model is not None
    # Check if subsequent calls return the same model instance
    assert TextRecognizer.get_model() is model


def test_audio_to_text_json_format(sample_audio_path, output_dir):
    result = TextRecognizer.audio_to_text(
        sample_audio_path, 
        output_format="json",
        output_dir=output_dir
    )
    assert isinstance(result, str)
    
    output_file = sample_audio_path.replace(".wav", ".json")

    assert os.path.exists(output_file)
    
    with open(output_file, 'r', encoding='utf-8') as f:
        content = json.load(f)
    assert "text" in content
    assert "segments" in content
    assert "timestamp" in content

def test_audio_to_text_srt_format(sample_audio_path, output_dir):
    result = TextRecognizer.audio_to_text(
        sample_audio_path, 
        output_format="srt",
        output_dir=output_dir
    )
    assert isinstance(result, str)
    
    output_file = sample_audio_path.replace(".wav", ".srt")
    assert os.path.exists(output_file)
    
    with open(output_file, 'r', encoding='utf-8') as f:
        content = f.read()
    assert "-->" in content  # Basic check for SRT format

def test_audio_to_text_error_handling():
    result = TextRecognizer.audio_to_text(
        "nonexistent_file.mp3",
        output_format="txt",
        output_dir="tests/data/output"
    )
    assert "Error processing audio" in result

