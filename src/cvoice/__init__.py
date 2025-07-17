"""
Cvoice: A comprehensive voice recognition and synthesis tool.

This package provides tools for transforming video audio from one person's voice
to another person's voice through a 5-step pipeline:
1. Audio extraction from video
2. Speech-to-text conversion
3. Text improvement and correction
4. Text-to-speech with voice cloning
5. Audio-video merging
"""

__version__ = "1.1.0"
__author__ = "isomo"
__email__ = "jiahaoxiang2000@gmail.com"

from .core.pipeline import VoiceClonePipeline

__all__ = ["VoiceClonePipeline"]


def main() -> None:
    """Main entry point for the CLI."""
    from .cli.main import main as cli_main
    cli_main()
