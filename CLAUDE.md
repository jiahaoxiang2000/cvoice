# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cvoice is a comprehensive voice recognition and synthesis tool that transforms video audio from one person's voice to another through a 5-step pipeline:

1. **Audio Extraction**: Extract audio from video files using MoviePy
2. **Speech-to-Text**: Convert audio to text using Whisper/SpeechBrain/Vosk
3. **Text Improvement**: Enhance transcription accuracy using AI models (OpenAI/Anthropic)
4. **Text-to-Speech**: Generate speech with voice cloning using Coqui TTS
5. **Video Merging**: Merge synthesized audio back into the original video

## Development Commands

### Package Management
```bash
# Install dependencies (UV is preferred)
uv sync
uv sync --dev  # Include development dependencies
```

### Testing
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_core/test_pipeline.py

# Run specific test class/method
uv run pytest tests/test_core/test_pipeline.py::TestVoiceClonePipeline::test_pipeline_initialization
```

### Code Quality
```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .
uv run ruff check --fix .  # Auto-fix issues

# Type checking
uv run mypy src/
```

### CLI Usage
```bash
# Main pipeline processing
uv run cvoice process input.mp4 --reference-audio reference.wav

# Individual steps
uv run cvoice extract-audio input.mp4 --output audio.wav
uv run cvoice transcribe audio.wav --language en
uv run cvoice synthesize "Hello world" --reference-audio ref.wav
```

## Architecture Overview

### Core Pipeline Architecture

The system follows a modular pipeline pattern where each component is a self-contained processor:

- **Base Classes** (`src/cvoice/core/base.py`): Abstract base classes defining the pipeline component interface
  - `PipelineComponent[T, U]`: Generic base class for all pipeline components
  - `AudioProcessor`, `TextProcessor`, `VideoProcessor`: Specialized base classes
  - `Pipeline`: Main orchestrator class

- **Main Pipeline** (`src/cvoice/core/pipeline.py`): Central orchestrator
  - `VoiceClonePipeline`: Main class that coordinates all components
  - `PipelineConfig`: Configuration dataclass for all pipeline settings
  - `PipelineResult`: Result dataclass containing processing outcomes

### CLI Architecture

The CLI (`src/cvoice/cli/main.py`) uses Click for command-line interface:

- **Commands**: `process`, `batch`, `extract-audio`, `transcribe`, `synthesize`, `info`
- **Rich Integration**: Progress bars, tables, and formatted output
- **Error Handling**: Consistent error reporting with debug mode support

## Configuration Management

## Testing Strategy

### Test Structure

Tests are organized by component in `tests/`:
- `tests/core/` - Core pipeline component tests
- `tests/utils/` - Utility function tests
- `tests/models/` - Model wrapper tests
- `tests/cli/` - CLI interface tests

### Test Fixtures

Common fixtures in `tests/conftest.py`:
- `temp_dir` - Temporary directory for test files
- `sample_audio_file` - Mock audio file for testing
- `sample_video_file` - Mock video file for testing

### Mocking Strategy

Heavy AI models and external APIs are mocked in tests:
- Mock model loading/unloading
- Mock API calls to OpenAI/Anthropic
- Mock file I/O operations for large media files

## Important Implementation Details

### Resource Management

All components use context managers for proper resource cleanup:
```python
with component:
    result = component.process(input_data)
```

### Error Handling

Consistent error handling pattern:
- Input validation in `validate_input()` method
- Component-specific errors wrapped in `RuntimeError`
- Logging at appropriate levels (debug, info, warning, error)

### File Management

- Unique filename generation to avoid conflicts
- Temporary file cleanup after processing
- Support for keeping intermediate files for debugging

### Performance Considerations

- Lazy loading of AI models
- Support for GPU acceleration (CUDA)
- Progress reporting for long-running operations

## Dependencies and External Requirements

### Core Dependencies
- **MoviePy**: Video/audio processing
- **Whisper/faster-whisper**: Speech recognition
- **Coqui TTS**: Text-to-speech synthesis
- **librosa/soundfile**: Audio analysis and I/O
- **Rich**: CLI formatting and progress bars

### Optional Dependencies
- **OpenAI/Anthropic**: For text improvement
- **PyTorch**: For GPU acceleration
- **FFmpeg**: Required by MoviePy for video processing
