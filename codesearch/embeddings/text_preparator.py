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
        Prepare a function for embedding with comprehensive error handling.

        Args:
            func: Function object with source code and docstring

        Returns:
            Prepared text combining docstring and source code
        """
        if func is None:
            return ""

        try:
            # Validate input
            if not func.source_code or not func.source_code.strip():
                # If no code, use docstring or function name as fallback
                if func.docstring and func.docstring.strip():
                    return func.docstring.strip()
                else:
                    return f"{func.name} function"

            # Apply preparation pipeline
            text = self._combine_text(func.docstring, func.source_code)
            text = self._filter_comments(text)
            text = self._truncate_to_tokens(text)

            if not text or not text.strip():
                # If all text was filtered out, fallback
                return func.docstring or f"{func.name} function"

            return text

        except Exception:
            # Log error but never fail completely
            # Try fallbacks in order
            if func.docstring and func.docstring.strip():
                return func.docstring.strip()
            return f"{func.name} function"

    def prepare_class(self, cls: Class) -> str:
        """
        Prepare a class for embedding with comprehensive error handling.

        Args:
            cls: Class object with source code and docstring

        Returns:
            Prepared text combining docstring and source code
        """
        if cls is None:
            return ""

        try:
            # Validate input
            if not cls.source_code or not cls.source_code.strip():
                # If no code, use docstring or class name as fallback
                if cls.docstring and cls.docstring.strip():
                    return cls.docstring.strip()
                else:
                    return f"{cls.name} class"

            # Apply preparation pipeline
            text = self._combine_text(cls.docstring, cls.source_code)
            text = self._filter_comments(text)
            text = self._truncate_to_tokens(text)

            if not text or not text.strip():
                # If all text was filtered out, fallback
                return cls.docstring or f"{cls.name} class"

            return text

        except Exception:
            # Log error but never fail completely
            # Try fallbacks in order
            if cls.docstring and cls.docstring.strip():
                return cls.docstring.strip()
            return f"{cls.name} class"

    def prepare_batch(self, items: List[Union[Function, Class]]) -> List[str]:
        """
        Prepare multiple items efficiently with error handling.

        Args:
            items: List of Function or Class objects

        Returns:
            List of prepared text strings
        """
        if not items:
            return []

        results = []
        for item in items:
            try:
                if isinstance(item, Function):
                    result = self.prepare_function(item)
                elif isinstance(item, Class):
                    result = self.prepare_class(item)
                else:
                    result = ""
                results.append(result)
            except Exception:
                # Continue processing others, append error fallback
                if isinstance(item, Function):
                    results.append(f"{item.name} function")
                elif isinstance(item, Class):
                    results.append(f"{item.name} class")
                else:
                    results.append("")

        return results

    def _filter_comments(self, code: str) -> str:
        """Filter code to keep important comments, remove trivial ones with error handling.

        Strategy:
        - Keep docstrings (already removed, not in code)
        - Keep block comments (consecutive # lines)
        - Keep inline comments with important keywords (TODO, FIXME, NOTE, BUG, HACK, WARNING, DEPRECATED, XXX)
        - Remove trivial inline comments without keywords
        """
        if not code or not code.strip():
            return code

        try:

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

        except Exception:
            # On error, return original code
            return code

    def _combine_text(self, docstring: Optional[str], source_code: str) -> str:
        """Combine docstring and source code with clear separation."""
        if docstring:
            return f"{docstring}\n\n{source_code}"
        return source_code

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the embedding model's tokenizer."""
        if not text or not text.strip():
            return 0
        try:
            tokens = self.tokenizer.encode(text)
            return len(tokens)
        except Exception:
            # Fallback: rough estimate (1 token ≈ 4 characters)
            return max(1, len(text) // 4)

    def _truncate_to_tokens(self, text: str) -> str:
        """Truncate text to stay within token limit with error handling.

        Strategies (in order):
        1. Keep all text if under limit
        2. Keep first N lines if over limit
        3. Keep docstring/signature only as fallback
        """
        if not text or not text.strip():
            return text

        try:
            token_count = self._count_tokens(text)
            if token_count <= self.max_tokens:
                return text

            # Strategy 1: Keep first N lines
            lines = text.split('\n')
            for i in range(len(lines), 0, -1):
                truncated = '\n'.join(lines[:i])
                if self._count_tokens(truncated) <= self.max_tokens:
                    return truncated

            # Strategy 2: Keep just first line (function signature or docstring)
            if lines:
                first_line = lines[0]
                if self._count_tokens(first_line) <= self.max_tokens:
                    return first_line

            # Strategy 3: Return empty string (last resort)
            return ""

        except Exception:
            # On error, return original (better to exceed limit than lose text)
            return text
