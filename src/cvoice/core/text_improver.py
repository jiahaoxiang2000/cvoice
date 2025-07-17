"""Text improvement and correction module."""

import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..utils.file_utils import ensure_dir, get_unique_filename
from .base import TextProcessor


@dataclass
class TextImprovementResult:
    """Result of text improvement."""
    original_text: str
    improved_text: str
    corrections: list[dict[str, Any]]
    confidence: float
    model_used: str


class TextImprover(TextProcessor):
    """Improve and correct transcribed text."""

    def __init__(
        self,
        api_provider: str = "openai",
        api_key: str | None = None,
        model_name: str = "gpt-3.5-turbo",
        max_tokens: int = 1000,
        temperature: float = 0.1,
        output_dir: Path | None = None,
        **kwargs
    ) -> None:
        """Initialize text improver.
        
        Args:
            api_provider: API provider (openai, anthropic, local)
            api_key: API key for the provider
            model_name: Model name to use
            max_tokens: Maximum tokens for response
            temperature: Temperature for generation
            output_dir: Directory to save improvement results
            **kwargs: Additional configuration
        """
        super().__init__("TextImprover", **kwargs)
        self.api_provider = api_provider.lower()
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.output_dir = output_dir or Path(tempfile.gettempdir()) / "cvoice_improvements"

        ensure_dir(self.output_dir)

        # Validate API provider
        supported_providers = ["openai", "anthropic", "local"]
        if self.api_provider not in supported_providers:
            raise ValueError(f"Unsupported API provider: {api_provider}")

        # Load API key from environment if not provided
        if not self.api_key:
            import os
            if self.api_provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.api_provider == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY")

        if not self.api_key and self.api_provider != "local":
            raise ValueError(f"API key required for {self.api_provider}")

    def process(self, input_text: str) -> str:
        """Improve and correct text.
        
        Args:
            input_text: Text to improve
            
        Returns:
            Improved text
        """
        self.validate_input(input_text)

        self.logger.info(f"Improving text: {len(input_text)} characters")

        try:
            result = self.improve_text(input_text)

            # Save improvement result
            self._save_improvement_result(result)

            self.logger.info(f"Text improvement completed: {len(result.improved_text)} characters")
            return result.improved_text

        except Exception as e:
            self.logger.error(f"Text improvement failed: {e}")
            raise RuntimeError(f"Text improvement failed: {e}") from e

    def improve_text(self, text: str) -> TextImprovementResult:
        """Improve text using AI models.
        
        Args:
            text: Text to improve
            
        Returns:
            Text improvement result
        """
        self.validate_input(text)

        # Apply basic preprocessing
        preprocessed_text = self._preprocess_text(text)

        # Get AI-powered improvements
        improved_text, corrections = self._get_ai_improvements(preprocessed_text)

        # Apply post-processing
        final_text = self._postprocess_text(improved_text)

        # Calculate confidence score
        confidence = self._calculate_confidence(preprocessed_text, final_text, corrections)

        return TextImprovementResult(
            original_text=text,
            improved_text=final_text,
            corrections=corrections,
            confidence=confidence,
            model_used=f"{self.api_provider}:{self.model_name}"
        )

    def batch_improve(self, texts: list[str]) -> list[TextImprovementResult]:
        """Improve multiple texts.
        
        Args:
            texts: List of texts to improve
            
        Returns:
            List of improvement results
        """
        results = []

        for i, text in enumerate(texts):
            self.logger.info(f"Processing text {i+1}/{len(texts)}")

            try:
                result = self.improve_text(text)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to improve text {i+1}: {e}")
                # Return original text as fallback
                results.append(TextImprovementResult(
                    original_text=text,
                    improved_text=text,
                    corrections=[],
                    confidence=0.0,
                    model_used="fallback"
                ))

        return results

    def _preprocess_text(self, text: str) -> str:
        """Apply basic text preprocessing.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Fix common transcription errors
        text = re.sub(r'\buh\b', '', text)  # Remove filler words
        text = re.sub(r'\bum\b', '', text)
        text = re.sub(r'\bah\b', '', text)

        # Fix spacing around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([,.!?;:])\s*', r'\1 ', text)

        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text.strip())

        return text

    def _postprocess_text(self, text: str) -> str:
        """Apply post-processing to improved text.
        
        Args:
            text: Improved text
            
        Returns:
            Post-processed text
        """
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]

        # Ensure proper sentence endings
        if text and text[-1] not in '.!?':
            text += '.'

        return text

    def _get_ai_improvements(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        """Get AI-powered text improvements.
        
        Args:
            text: Text to improve
            
        Returns:
            Tuple of (improved_text, corrections)
        """
        if self.api_provider == "openai":
            return self._improve_with_openai(text)
        elif self.api_provider == "anthropic":
            return self._improve_with_anthropic(text)
        elif self.api_provider == "local":
            return self._improve_with_local_model(text)
        else:
            raise ValueError(f"Unsupported provider: {self.api_provider}")

    def _improve_with_openai(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        """Improve text using OpenAI API.
        
        Args:
            text: Text to improve
            
        Returns:
            Tuple of (improved_text, corrections)
        """
        try:
            import openai

            client = openai.OpenAI(api_key=self.api_key)

            prompt = self._create_improvement_prompt(text)

            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a text improvement assistant. Fix grammar, punctuation, and clarity while preserving the original meaning."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            improved_text = response.choices[0].message.content.strip()

            # Extract corrections (simplified)
            corrections = self._extract_corrections(text, improved_text)

            return improved_text, corrections

        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"OpenAI improvement failed: {e}") from e

    def _improve_with_anthropic(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        """Improve text using Anthropic API.
        
        Args:
            text: Text to improve
            
        Returns:
            Tuple of (improved_text, corrections)
        """
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)

            prompt = self._create_improvement_prompt(text)

            response = client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            improved_text = response.content[0].text.strip()

            # Extract corrections (simplified)
            corrections = self._extract_corrections(text, improved_text)

            return improved_text, corrections

        except Exception as e:
            self.logger.error(f"Anthropic API error: {e}")
            raise RuntimeError(f"Anthropic improvement failed: {e}") from e

    def _improve_with_local_model(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        """Improve text using local model.
        
        Args:
            text: Text to improve
            
        Returns:
            Tuple of (improved_text, corrections)
        """
        # Placeholder for local model implementation
        # This could use transformers library or local inference
        self.logger.warning("Local model improvement not implemented, returning original text")
        return text, []

    def _create_improvement_prompt(self, text: str) -> str:
        """Create improvement prompt for AI models.
        
        Args:
            text: Text to improve
            
        Returns:
            Improvement prompt
        """
        return f"""
Please improve the following transcribed text by:
1. Fixing grammar and punctuation errors
2. Correcting spelling mistakes
3. Improving clarity and readability
4. Removing filler words and unnecessary repetitions
5. Ensuring proper sentence structure

Original text: "{text}"

Please provide only the improved text without any additional commentary or explanation.
"""

    def _extract_corrections(self, original: str, improved: str) -> list[dict[str, Any]]:
        """Extract corrections made to the text.
        
        Args:
            original: Original text
            improved: Improved text
            
        Returns:
            List of corrections
        """
        # Simple diff-based correction extraction
        corrections = []

        # This is a simplified implementation
        # In practice, you'd use a proper diff algorithm
        if original != improved:
            corrections.append({
                "type": "general_improvement",
                "original": original,
                "corrected": improved,
                "confidence": 0.8
            })

        return corrections

    def _calculate_confidence(
        self,
        original: str,
        improved: str,
        corrections: list[dict[str, Any]]
    ) -> float:
        """Calculate confidence score for improvements.
        
        Args:
            original: Original text
            improved: Improved text
            corrections: List of corrections
            
        Returns:
            Confidence score (0-1)
        """
        # Simple confidence calculation
        if not corrections:
            return 1.0

        # Base confidence on number of corrections relative to text length
        correction_ratio = len(corrections) / max(len(original.split()), 1)

        # Higher confidence for fewer corrections
        confidence = max(0.1, 1.0 - correction_ratio)

        return confidence

    def _save_improvement_result(self, result: TextImprovementResult) -> None:
        """Save improvement result to file.
        
        Args:
            result: Improvement result
        """
        try:
            output_filename = "text_improvement_result.json"
            output_path = self.output_dir / output_filename

            # Ensure unique filename
            output_path = get_unique_filename(output_path.with_suffix(""), "json")

            result_dict = {
                "original_text": result.original_text,
                "improved_text": result.improved_text,
                "corrections": result.corrections,
                "confidence": result.confidence,
                "model_used": result.model_used
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)

            self.logger.debug(f"Improvement result saved: {output_path}")

        except Exception as e:
            self.logger.warning(f"Failed to save improvement result: {e}")

    def get_supported_models(self) -> list[str]:
        """Get list of supported models for the current provider.
        
        Returns:
            List of supported model names
        """
        models = {
            "openai": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            "anthropic": ["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"],
            "local": ["local-model"]
        }

        return models.get(self.api_provider, [])
