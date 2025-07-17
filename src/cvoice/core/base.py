"""Base classes for pipeline components."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar

T = TypeVar('T')
U = TypeVar('U')


class PipelineComponent(ABC, Generic[T, U]):
    """Base class for pipeline components."""

    def __init__(self, name: str, **kwargs: Any) -> None:
        """Initialize the component.
        
        Args:
            name: Name of the component
            **kwargs: Additional configuration parameters
        """
        self.name = name
        self.config = kwargs
        self.logger = logging.getLogger(f"cvoice.{name}")

    @abstractmethod
    def process(self, input_data: T) -> U:
        """Process input data and return output.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Processed output data
        """
        pass

    def validate_input(self, input_data: T) -> None:
        """Validate input data.
        
        Args:
            input_data: Input data to validate
            
        Raises:
            ValueError: If input data is invalid
        """
        if input_data is None:
            raise ValueError(f"{self.name}: Input data cannot be None")

    def setup(self) -> None:
        """Set up the component (called before processing)."""
        self.logger.info(f"Setting up {self.name}")

    def teardown(self) -> None:
        """Tear down the component (called after processing)."""
        self.logger.info(f"Tearing down {self.name}")

    def __enter__(self) -> "PipelineComponent[T, U]":
        """Context manager entry."""
        self.setup()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.teardown()


class AudioProcessor(PipelineComponent[Path, Path]):
    """Base class for audio processing components."""

    def validate_input(self, input_path: Path) -> None:
        """Validate audio input file.
        
        Args:
            input_path: Path to audio file
            
        Raises:
            ValueError: If file is invalid
        """
        super().validate_input(input_path)

        if not input_path.exists():
            raise ValueError(f"{self.name}: Input file does not exist: {input_path}")

        if not input_path.is_file():
            raise ValueError(f"{self.name}: Input path is not a file: {input_path}")


class TextProcessor(PipelineComponent[str, str]):
    """Base class for text processing components."""

    def validate_input(self, input_text: str) -> None:
        """Validate text input.
        
        Args:
            input_text: Input text to validate
            
        Raises:
            ValueError: If text is invalid
        """
        super().validate_input(input_text)

        if not input_text.strip():
            raise ValueError(f"{self.name}: Input text cannot be empty")


class VideoProcessor(PipelineComponent[Path, Path]):
    """Base class for video processing components."""

    def validate_input(self, input_path: Path) -> None:
        """Validate video input file.
        
        Args:
            input_path: Path to video file
            
        Raises:
            ValueError: If file is invalid
        """
        super().validate_input(input_path)

        if not input_path.exists():
            raise ValueError(f"{self.name}: Input file does not exist: {input_path}")

        if not input_path.is_file():
            raise ValueError(f"{self.name}: Input path is not a file: {input_path}")

        # Check for common video extensions
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        if input_path.suffix.lower() not in video_extensions:
            self.logger.warning(f"Unusual file extension: {input_path.suffix}")


class Pipeline:
    """Main pipeline orchestrator."""

    def __init__(self, name: str) -> None:
        """Initialize the pipeline.
        
        Args:
            name: Name of the pipeline
        """
        self.name = name
        self.components: list[PipelineComponent[Any, Any]] = []
        self.logger = logging.getLogger(f"cvoice.pipeline.{name}")

    def add_component(self, component: PipelineComponent[Any, Any]) -> None:
        """Add a component to the pipeline.
        
        Args:
            component: Component to add
        """
        self.components.append(component)
        self.logger.info(f"Added component: {component.name}")

    def run(self, input_data: Any) -> Any:
        """Run the pipeline with input data.
        
        Args:
            input_data: Initial input data
            
        Returns:
            Final output data
        """
        self.logger.info(f"Starting pipeline: {self.name}")

        current_data = input_data

        for i, component in enumerate(self.components):
            self.logger.info(f"Processing step {i+1}/{len(self.components)}: {component.name}")

            try:
                with component:
                    current_data = component.process(current_data)
            except Exception as e:
                self.logger.error(f"Error in {component.name}: {e}")
                raise

        self.logger.info(f"Pipeline completed: {self.name}")
        return current_data
