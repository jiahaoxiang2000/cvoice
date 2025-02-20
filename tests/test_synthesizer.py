import os
import pytest
from ..audio.synthesizer import AudioSynthesizer
from ..utils.file_handler import FileHandler


@pytest.fixture
def audio_synthesizer():
    return AudioSynthesizer()


def test_text_to_audio_custom_path(audio_synthesizer):
    text = "Testing custom output path, 这是一个测试"
    custom_path = os.path.join("../data", "text2audio.wav")
    
    output_path = audio_synthesizer.text_to_audio(text, custom_path)
    
    assert output_path == custom_path
    assert os.path.exists(custom_path)
    assert os.path.getsize(custom_path) > 0



