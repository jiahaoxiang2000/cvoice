"""Tests for the main pipeline."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from cvoice.core.pipeline import VoiceClonePipeline, PipelineConfig


class TestPipelineConfig:
    """Test pipeline configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PipelineConfig()
        
        assert config.audio_format == "wav"
        assert config.audio_sample_rate == 22050
        assert config.stt_model == "base"
        assert config.text_improvement_enabled is True
        assert config.tts_model == "tts_models/multilingual/multi-dataset/xtts_v2"
        assert config.video_format == "mp4"
        
    def test_custom_config(self):
        """Test custom configuration values."""
        config = PipelineConfig(
            audio_format="mp3",
            stt_model="large",
            text_improvement_enabled=False
        )
        
        assert config.audio_format == "mp3"
        assert config.stt_model == "large"
        assert config.text_improvement_enabled is False


class TestVoiceClonePipeline:
    """Test voice cloning pipeline."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = VoiceClonePipeline()
        
        assert pipeline.config is not None
        assert pipeline.audio_extractor is not None
        assert pipeline.speech_to_text is not None
        assert pipeline.text_to_speech is not None
        assert pipeline.video_merger is not None
        
    def test_pipeline_with_custom_config(self):
        """Test pipeline with custom configuration."""
        config = PipelineConfig(
            stt_model="large",
            text_improvement_enabled=False
        )
        pipeline = VoiceClonePipeline(config)
        
        assert pipeline.config.stt_model == "large"
        assert pipeline.config.text_improvement_enabled is False
        assert pipeline.text_improver is None
        
    @patch('cvoice.core.pipeline.VideoExtractor')
    @patch('cvoice.core.pipeline.SpeechToText')
    @patch('cvoice.core.pipeline.TextToSpeech')
    @patch('cvoice.core.pipeline.VideoMerger')
    def test_process_video_success(self, mock_merger, mock_tts, mock_stt, mock_extractor):
        """Test successful video processing."""
        # Mock component outputs
        mock_extractor.return_value.process.return_value = Path("extracted_audio.wav")
        mock_stt.return_value.process.return_value = "Hello world"
        mock_tts.return_value.process.return_value = Path("synthesized_audio.wav")
        mock_merger.return_value.process.return_value = Path("output_video.mp4")
        
        pipeline = VoiceClonePipeline()
        
        # Mock the actual components
        pipeline.audio_extractor = mock_extractor.return_value
        pipeline.speech_to_text = mock_stt.return_value
        pipeline.text_to_speech = mock_tts.return_value
        pipeline.video_merger = mock_merger.return_value
        
        result = pipeline.process_video(
            input_video=Path("test_video.mp4"),
            reference_audio=Path("reference.wav")
        )
        
        assert result.success is True
        assert result.transcribed_text == "Hello world"
        assert result.improved_text == "Hello world"  # No improvement when disabled
        
    def test_get_pipeline_info(self):
        """Test pipeline information retrieval."""
        pipeline = VoiceClonePipeline()
        info = pipeline.get_pipeline_info()
        
        assert "config" in info
        assert "components" in info
        assert "output_dir" in info
        assert info["components"]["audio_extractor"] == "AudioExtractor"