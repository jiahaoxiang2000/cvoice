import pytest
import os
from ..audio.synthesizer import AudioSynthesizer

def test_text_to_audio_creates_file(sample_text):
    try:
        output_path = AudioSynthesizer.text_to_audio(sample_text)
        assert os.path.exists(output_path)
        assert output_path.endswith('.wav')
    except RuntimeError as e:
        if "say: command not found" in str(e):
            pytest.skip("'say' command not available on this system")

def test_text_to_audio_invalid_input():
    with pytest.raises(RuntimeError):
        AudioSynthesizer.text_to_audio("")
