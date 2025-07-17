"""Command-line interface for Cvoice."""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ..core.pipeline import PipelineConfig, VoiceClonePipeline
from ..utils.file_utils import validate_file_path
from ..utils.logging_utils import setup_logging

console = Console()


@click.group()
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.option('--log-file', type=click.Path(), help='Log file path')
@click.pass_context
def cli(ctx: click.Context, debug: bool, log_file: str | None) -> None:
    """Cvoice - Voice recognition and synthesis tool."""
    # Set up logging
    log_level = 'DEBUG' if debug else 'INFO'
    setup_logging(
        level=getattr(__import__('logging'), log_level),
        log_file=Path(log_file) if log_file else None
    )

    # Store context
    ctx.ensure_object(dict)
    ctx.obj['debug'] = debug


@cli.command()
@click.argument('input_video', type=click.Path(exists=True, path_type=Path))
@click.option('--reference-audio', '-r', type=click.Path(exists=True, path_type=Path),
              help='Reference audio file for voice cloning')
@click.option('--output', '-o', type=click.Path(path_type=Path),
              help='Output video file path')
@click.option('--language', '-l', default='en', help='Language code for processing')
@click.option('--stt-model', default='base', help='Speech-to-text model')
@click.option('--tts-model', default='tts_models/multilingual/multi-dataset/xtts_v2',
              help='Text-to-speech model')
@click.option('--no-text-improvement', is_flag=True,
              help='Skip text improvement step')
@click.option('--keep-intermediate', is_flag=True,
              help='Keep intermediate files')
@click.option('--device', default='auto', help='Device to use (cpu/cuda/auto)')
@click.pass_context
def process(
    ctx: click.Context,
    input_video: Path,
    reference_audio: Path | None,
    output: Path | None,
    language: str,
    stt_model: str,
    tts_model: str,
    no_text_improvement: bool,
    keep_intermediate: bool,
    device: str
) -> None:
    """Process a video file through the voice cloning pipeline."""
    try:
        # Validate input
        validate_file_path(input_video)
        if reference_audio:
            validate_file_path(reference_audio)

        # Create pipeline configuration
        config = PipelineConfig(
            stt_model=stt_model,
            stt_language=language,
            stt_device=device,
            text_improvement_enabled=not no_text_improvement,
            tts_model=tts_model,
            tts_language=language,
            tts_device=device,
            keep_intermediate_files=keep_intermediate
        )

        # Initialize pipeline
        pipeline = VoiceClonePipeline(config)

        # Show processing info
        info_panel = Panel(
            f"[bold]Input Video:[/bold] {input_video}\n"
            f"[bold]Reference Audio:[/bold] {reference_audio or 'None'}\n"
            f"[bold]Language:[/bold] {language}\n"
            f"[bold]STT Model:[/bold] {stt_model}\n"
            f"[bold]TTS Model:[/bold] {tts_model}\n"
            f"[bold]Device:[/bold] {device}",
            title="Processing Configuration"
        )
        console.print(info_panel)

        # Process video with progress indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Processing video...", total=None)

            result = pipeline.process_video(
                input_video=input_video,
                reference_audio=reference_audio,
                output_path=output
            )

            progress.update(task, completed=True)

        # Display results
        if result.success:
            console.print("✅ [green]Processing completed successfully![/green]")
            console.print(f"📁 Output video: {result.output_video}")
            console.print(f"⏱️  Processing time: {result.processing_time:.2f} seconds")

            # Show transcription preview
            if result.transcribed_text:
                text_preview = result.transcribed_text[:200]
                if len(result.transcribed_text) > 200:
                    text_preview += "..."
                console.print(f"📝 Transcription: {text_preview}")

        else:
            console.print(f"❌ [red]Processing failed: {result.error_message}[/red]")
            sys.exit(1)

    except Exception as e:
        console.print(f"❌ [red]Error: {e}[/red]")
        if ctx.obj.get('debug'):
            raise
        sys.exit(1)


@cli.command()
@click.argument('input_videos', nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option('--reference-audio', '-r', type=click.Path(exists=True, path_type=Path),
              help='Reference audio file for voice cloning')
@click.option('--output-dir', '-d', type=click.Path(path_type=Path),
              help='Output directory for processed videos')
@click.option('--language', '-l', default='en', help='Language code for processing')
@click.option('--stt-model', default='base', help='Speech-to-text model')
@click.option('--tts-model', default='tts_models/multilingual/multi-dataset/xtts_v2',
              help='Text-to-speech model')
@click.option('--no-text-improvement', is_flag=True,
              help='Skip text improvement step')
@click.option('--keep-intermediate', is_flag=True,
              help='Keep intermediate files')
@click.option('--device', default='auto', help='Device to use (cpu/cuda/auto)')
@click.pass_context
def batch(
    ctx: click.Context,
    input_videos: tuple[Path, ...],
    reference_audio: Path | None,
    output_dir: Path | None,
    language: str,
    stt_model: str,
    tts_model: str,
    no_text_improvement: bool,
    keep_intermediate: bool,
    device: str
) -> None:
    """Process multiple video files in batch."""
    if not input_videos:
        console.print("❌ [red]No input videos provided[/red]")
        sys.exit(1)

    try:
        # Validate inputs
        video_paths = list(input_videos)
        for video_path in video_paths:
            validate_file_path(video_path)

        if reference_audio:
            validate_file_path(reference_audio)

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        # Create pipeline configuration
        config = PipelineConfig(
            stt_model=stt_model,
            stt_language=language,
            stt_device=device,
            text_improvement_enabled=not no_text_improvement,
            tts_model=tts_model,
            tts_language=language,
            tts_device=device,
            keep_intermediate_files=keep_intermediate
        )

        # Initialize pipeline
        pipeline = VoiceClonePipeline(config)

        # Show batch info
        info_panel = Panel(
            f"[bold]Input Videos:[/bold] {len(video_paths)}\n"
            f"[bold]Reference Audio:[/bold] {reference_audio or 'None'}\n"
            f"[bold]Output Directory:[/bold] {output_dir or 'Default'}\n"
            f"[bold]Language:[/bold] {language}\n"
            f"[bold]STT Model:[/bold] {stt_model}\n"
            f"[bold]TTS Model:[/bold] {tts_model}",
            title="Batch Processing Configuration"
        )
        console.print(info_panel)

        # Process videos
        with Progress(console=console) as progress:
            task = progress.add_task("Processing videos...", total=len(video_paths))

            results = []
            for i, video_path in enumerate(video_paths):
                progress.update(task, description=f"Processing {video_path.name}...")

                result = pipeline.process_video(
                    input_video=video_path,
                    reference_audio=reference_audio,
                    output_path=output_dir / f"{video_path.stem}_voice_cloned.mp4" if output_dir else None
                )

                results.append(result)
                progress.advance(task)

        # Display results summary
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]

        console.print("\n📊 [bold]Batch Processing Summary[/bold]")
        console.print(f"✅ Successful: {len(successful_results)}")
        console.print(f"❌ Failed: {len(failed_results)}")

        if failed_results:
            console.print("\n[red]Failed videos:[/red]")
            for result in failed_results:
                console.print(f"  - {result.input_video}: {result.error_message}")

    except Exception as e:
        console.print(f"❌ [red]Batch processing error: {e}[/red]")
        if ctx.obj.get('debug'):
            raise
        sys.exit(1)


@cli.command()
@click.argument('input_video', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o', type=click.Path(path_type=Path),
              help='Output audio file path')
@click.option('--format', '-f', default='wav', help='Audio format')
@click.option('--sample-rate', '-s', default=22050, help='Sample rate')
def extract_audio(
    input_video: Path,
    output: Path | None,
    format: str,
    sample_rate: int
) -> None:
    """Extract audio from video file."""
    try:
        from ..core.audio_extractor import AudioExtractor

        extractor = AudioExtractor(
            output_format=format,
            sample_rate=sample_rate
        )

        with Progress(console=console) as progress:
            task = progress.add_task("Extracting audio...", total=None)

            result = extractor.process(input_video)

            progress.update(task, completed=True)

        if output and result != output:
            result.rename(output)
            result = output

        console.print(f"✅ Audio extracted: {result}")

    except Exception as e:
        console.print(f"❌ [red]Audio extraction failed: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('input_audio', type=click.Path(exists=True, path_type=Path))
@click.option('--language', '-l', help='Language code')
@click.option('--model', '-m', default='base', help='Whisper model')
@click.option('--device', default='auto', help='Device to use')
@click.option('--output', '-o', type=click.Path(path_type=Path),
              help='Output text file path')
def transcribe(
    input_audio: Path,
    language: str | None,
    model: str,
    device: str,
    output: Path | None
) -> None:
    """Transcribe audio to text."""
    try:
        from ..core.speech_to_text import SpeechToText

        stt = SpeechToText(
            model_name=model,
            device=device,
            language=language
        )

        with Progress(console=console) as progress:
            task = progress.add_task("Transcribing audio...", total=None)

            with stt:
                text = stt.process(input_audio)

            progress.update(task, completed=True)

        if output:
            output.write_text(text, encoding='utf-8')
            console.print(f"✅ Transcription saved: {output}")
        else:
            console.print(f"📝 Transcription:\n{text}")

    except Exception as e:
        console.print(f"❌ [red]Transcription failed: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('input_text', type=str)
@click.option('--reference-audio', '-r', type=click.Path(exists=True, path_type=Path),
              help='Reference audio file for voice cloning')
@click.option('--output', '-o', type=click.Path(path_type=Path),
              help='Output audio file path')
@click.option('--language', '-l', default='en', help='Language code')
@click.option('--model', '-m', default='tts_models/multilingual/multi-dataset/xtts_v2',
              help='TTS model')
@click.option('--device', default='auto', help='Device to use')
def synthesize(
    input_text: str,
    reference_audio: Path | None,
    output: Path | None,
    language: str,
    model: str,
    device: str
) -> None:
    """Synthesize speech from text."""
    try:
        from ..core.text_to_speech import TextToSpeech

        tts = TextToSpeech(
            model_name=model,
            device=device,
            language=language,
            reference_audio=reference_audio
        )

        with Progress(console=console) as progress:
            task = progress.add_task("Synthesizing speech...", total=None)

            with tts:
                result = tts.process(input_text)

            progress.update(task, completed=True)

        if output and result != output:
            result.rename(output)
            result = output

        console.print(f"✅ Speech synthesized: {result}")

    except Exception as e:
        console.print(f"❌ [red]Speech synthesis failed: {e}[/red]")
        sys.exit(1)


@cli.command()
def info() -> None:
    """Show system and model information."""
    try:
        import torch

        from ..core.pipeline import VoiceClonePipeline

        # System info
        table = Table(title="System Information")
        table.add_column("Component", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Python Version", sys.version.split()[0])
        table.add_row("PyTorch Version", torch.__version__)
        table.add_row("CUDA Available", str(torch.cuda.is_available()))
        if torch.cuda.is_available():
            table.add_row("CUDA Device", torch.cuda.get_device_name(0))

        console.print(table)

        # Pipeline info
        pipeline = VoiceClonePipeline()
        pipeline_info = pipeline.get_pipeline_info()

        config_table = Table(title="Pipeline Configuration")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="green")

        for key, value in pipeline_info["config"].items():
            config_table.add_row(key, str(value))

        console.print(config_table)

    except Exception as e:
        console.print(f"❌ [red]Failed to get system info: {e}[/red]")
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == '__main__':
    main()
