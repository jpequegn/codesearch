"""Text preparation for code embeddings."""

from typing import List, Optional, Union

from transformers import AutoTokenizer

from codesearch.models import Function, Class


class TextPreparator:
    """Prepares code text for optimal embedding generation."""

    def __init__(self, tokenizer: AutoTokenizer, max_tokens: int = 512):
        """
        Initialize the text preparator.

        Args:
            tokenizer: HuggingFace tokenizer for token counting
            max_tokens: Maximum tokens to keep in prepared text
        """
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def prepare_function(self, func: Function) -> str:
        """
        Prepare a function for embedding.

        Args:
            func: Function object with source code and docstring

        Returns:
            Prepared text combining docstring and source code
        """
        return self._combine_text(func.docstring, func.source_code)

    def prepare_class(self, cls: Class) -> str:
        """
        Prepare a class for embedding.

        Args:
            cls: Class object with source code and docstring

        Returns:
            Prepared text combining docstring and source code
        """
        return self._combine_text(cls.docstring, cls.source_code)

    def prepare_batch(self, items: List[Union[Function, Class]]) -> List[str]:
        """
        Prepare multiple items for embedding.

        Args:
            items: List of Function or Class objects

        Returns:
            List of prepared text strings
        """
        return [
            self.prepare_function(item) if isinstance(item, Function)
            else self.prepare_class(item)
            for item in items
        ]

    def _combine_text(self, docstring: Optional[str], source_code: str) -> str:
        """Combine docstring and source code with clear separation."""
        if docstring:
            return f"{docstring}\n\n{source_code}"
        return source_code
