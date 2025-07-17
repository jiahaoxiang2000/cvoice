"""Base classes for AI models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseModel(ABC):
    """Base class for AI models."""

    def __init__(self, model_path: Path | None = None, **kwargs: Any) -> None:
        """Initialize the model.
        
        Args:
            model_path: Path to the model file
            **kwargs: Additional model-specific parameters
        """
        self.model_path = model_path
        self.config = kwargs
        self._model: Any | None = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory."""
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """Unload the model from memory."""
        pass

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None

    def __enter__(self) -> "BaseModel":
        """Context manager entry."""
        self.load_model()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.unload_model()
