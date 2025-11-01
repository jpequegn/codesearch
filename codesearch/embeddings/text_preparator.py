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
        try:
            text = self._combine_text(func.docstring, func.source_code)
            text = self._filter_comments(text)
            # Token counting and truncation will be added in Task 4
            return text
        except Exception:
            # Fallback will be added in Task 5
            return func.docstring or f"{func.name} function"

    def prepare_class(self, cls: Class) -> str:
        """
        Prepare a class for embedding.

        Args:
            cls: Class object with source code and docstring

        Returns:
            Prepared text combining docstring and source code
        """
        try:
            text = self._combine_text(cls.docstring, cls.source_code)
            text = self._filter_comments(text)
            # Token counting and truncation will be added in Task 4
            return text
        except Exception:
            # Fallback will be added in Task 5
            return cls.docstring or f"{cls.name} class"

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

    def _filter_comments(self, code: str) -> str:
        """Filter code to keep important comments, remove trivial ones.

        Strategy:
        - Keep docstrings (already removed, not in code)
        - Keep block comments (consecutive # lines)
        - Keep inline comments with important keywords (TODO, FIXME, NOTE, BUG, HACK, WARNING, DEPRECATED, XXX)
        - Remove trivial inline comments without keywords
        """
        import re

        IMPORTANT_KEYWORDS = {
            'TODO', 'FIXME', 'NOTE', 'BUG', 'HACK',
            'WARNING', 'DEPRECATED', 'XXX'
        }

        lines = code.split('\n')
        filtered_lines = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Check if this is a block comment (starts with # after stripping)
            stripped = line.lstrip()
            if stripped.startswith('#'):
                # Look ahead to see if this is part of a block
                is_block = False
                if i + 1 < len(lines):
                    next_stripped = lines[i + 1].lstrip()
                    is_block = next_stripped.startswith('#')

                # Check if comment contains important keyword
                has_important = any(kw in line.upper() for kw in IMPORTANT_KEYWORDS)

                if is_block or has_important:
                    # Keep block comments and important comments
                    filtered_lines.append(line)
                else:
                    # Skip trivial single-line comment
                    pass
            else:
                # This is code (not a comment line)
                # Check for inline comments
                if '#' in line:
                    # Split on first # outside of strings
                    parts = line.split('#', 1)
                    code_part = parts[0]
                    comment_part = '#' + parts[1] if len(parts) > 1 else ''

                    # Check if inline comment has important keyword
                    has_important = any(kw in comment_part.upper() for kw in IMPORTANT_KEYWORDS)

                    if has_important:
                        # Keep the code and the important comment
                        filtered_lines.append(line)
                    else:
                        # Keep only the code part
                        filtered_lines.append(code_part.rstrip())
                else:
                    # No comment in this line, keep as is
                    filtered_lines.append(line)

            i += 1

        return '\n'.join(filtered_lines)

    def _combine_text(self, docstring: Optional[str], source_code: str) -> str:
        """Combine docstring and source code with clear separation."""
        if docstring:
            return f"{docstring}\n\n{source_code}"
        return source_code
