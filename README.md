# Cvoice

A comprehensive voice recognition and synthesis tool that transforms video audio from one person's voice to another person's voice through a sophisticated 5-step pipeline.

## Features

- **Audio Extraction**: Extract high-quality audio from video files
- **Speech-to-Text**: Convert speech to text using state-of-the-art models (Whisper, SpeechBrain, Vosk)
- **Text Improvement**: Enhance transcribed text for better accuracy using AI models
- **Voice Cloning**: Generate speech with target voice characteristics using advanced TTS models
- **Video Merging**: Seamlessly merge synthesized audio back into the original video

## How it Works

1. **Audio Extraction**: Split audio from video files while preserving quality
2. **Speech Recognition**: Convert audio to text using advanced ASR models
3. **Text Enhancement**: Improve transcription accuracy and fix errors using AI
4. **Voice Synthesis**: Generate speech with cloned voice characteristics
5. **Video Integration**: Merge the new audio track back into the original video

## Installation

### Requirements

- Python 3.11 or higher
- UV package manager (recommended) or pip
- FFmpeg (for video/audio processing)

### Install with UV (Recommended)

```bash
git clone https://github.com/your-username/cvoice.git
cd cvoice
uv sync
```

### Install with pip

```bash
git clone https://github.com/your-username/cvoice.git
cd cvoice
pip install -e .
```

## Quick Start

### Basic Usage

```bash
# Process a video with voice cloning
cvoice process input_video.mp4 --reference-audio reference_voice.wav

# Extract audio from video
cvoice extract-audio input_video.mp4 --output extracted_audio.wav

# Transcribe audio to text
cvoice transcribe audio_file.wav --language en

# Synthesize speech from text
cvoice synthesize "Hello, this is a test" --reference-audio voice_sample.wav
```

### Advanced Usage

```bash
# Process with custom models and settings
cvoice process video.mp4 \
  --reference-audio reference.wav \
  --stt-model large \
  --tts-model tts_models/multilingual/multi-dataset/xtts_v2 \
  --language en \
  --device cuda \
  --keep-intermediate

# Batch process multiple videos
cvoice batch video1.mp4 video2.mp4 video3.mp4 \
  --reference-audio reference.wav \
  --output-dir ./processed_videos
```

## Python API

```python
from cvoice import VoiceClonePipeline, PipelineConfig
from pathlib import Path

# Create pipeline configuration
config = PipelineConfig(
    stt_model="base",
    tts_model="tts_models/multilingual/multi-dataset/xtts_v2",
    language="en",
    text_improvement_enabled=True
)

# Initialize pipeline
pipeline = VoiceClonePipeline(config)

# Process video
result = pipeline.process_video(
    input_video=Path("input.mp4"),
    reference_audio=Path("reference.wav"),
    output_path=Path("output.mp4")
)

if result.success:
    print(f"Processing completed: {result.output_video}")
    print(f"Transcription: {result.transcribed_text}")
else:
    print(f"Processing failed: {result.error_message}")
```

## Configuration

### Speech-to-Text Models

- **Whisper**: `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3`
- **SpeechBrain**: Various pre-trained models
- **Vosk**: Lightweight offline models
- **Deepgram**: Cloud-based API (requires API key)

### Text-to-Speech Models

- **Coqui TTS**: `tts_models/multilingual/multi-dataset/xtts_v2` (recommended)
- **Custom Models**: Support for custom TTS models

### Text Improvement

- **OpenAI**: GPT-3.5, GPT-4 (requires API key)
- **Anthropic**: Claude models (requires API key)
- **Local Models**: Support for local language models

## Environment Variables

```bash
# API keys for text improvement
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export DEEPGRAM_API_KEY="your-deepgram-api-key"
```

## Performance Tips

1. **GPU Acceleration**: Use CUDA for faster processing
2. **Model Selection**: Choose appropriate model size for your hardware
3. **Batch Processing**: Process multiple videos together for efficiency
4. **Reference Audio**: Use high-quality reference audio (5+ seconds)

## Supported Formats

### Input Formats
- **Video**: MP4, AVI, MOV, MKV, WMV, FLV, WEBM
- **Audio**: WAV, MP3, M4A, FLAC, OGG, AAC

### Output Formats
- **Video**: MP4, AVI, MOV
- **Audio**: WAV, MP3, M4A, FLAC

## CLI Commands

### Main Commands

- `cvoice process` - Process single video through complete pipeline
- `cvoice batch` - Process multiple videos
- `cvoice extract-audio` - Extract audio from video
- `cvoice transcribe` - Convert audio to text
- `cvoice synthesize` - Generate speech from text
- `cvoice info` - Show system and model information

### Global Options

- `--debug` - Enable debug logging
- `--log-file` - Specify log file path
- `--help` - Show help information

## Development

### Setting up Development Environment

```bash
# Clone repository
git clone https://github.com/your-username/cvoice.git
cd cvoice

# Install with development dependencies
uv sync --dev

# Install pre-commit hooks
uv run pre-commit install
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_core/test_pipeline.py
```

### Code Quality

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Type checking
uv run mypy src/
```

## Architecture

### Core Components

1. **AudioExtractor**: Extracts audio from video files
2. **SpeechToText**: Converts audio to text using various ASR models
3. **TextImprover**: Enhances transcribed text using AI models
4. **TextToSpeech**: Synthesizes speech with voice cloning
5. **VideoMerger**: Merges synthesized audio back into video

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run tests and ensure they pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI Whisper for speech recognition
- Coqui TTS for text-to-speech synthesis
- MoviePy for video/audio processing
- FastAPI for potential web interface
- Rich for beautiful CLI output

## Support

For issues, questions, or contributions, please visit our [GitHub repository](https://github.com/your-username/cvoice).