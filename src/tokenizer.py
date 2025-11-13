"""
OpenAI tokenizer wrapper for compatibility with HybridChunker.
"""

from typing import Dict, List, Tuple
from tiktoken import get_encoding
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class OpenAITokenizerWrapper(PreTrainedTokenizerBase):
    """Minimal wrapper for OpenAI's tokenizer to work with HybridChunker."""

    def __init__(
        self, model_name: str = "cl100k_base", max_length: int = 8191, **kwargs
    ):
        """
        Initialize the tokenizer.

        Args:
            model_name: The name of the OpenAI encoding to use (default: cl100k_base for gpt-4/gpt-3.5/text-embedding-3)
            max_length: Maximum sequence length
        """
        super().__init__(model_max_length=max_length, **kwargs)
        self.tokenizer = get_encoding(model_name)
        self._vocab_size = self.tokenizer.max_token_value

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """Main method used by HybridChunker."""
        return [str(t) for t in self.tokenizer.encode(text)]

    def _tokenize(self, text: str) -> List[str]:
        return self.tokenize(text)

    def encode(self, text: str, add_special_tokens: bool = True, **kwargs) -> List[int]:
        """
        Encode text into token IDs.

        Args:
            text: Text to encode
            add_special_tokens: Whether to add special tokens (ignored for tiktoken)
            **kwargs: Additional arguments (ignored)

        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(text)

    def encode_plus(self, text: str, add_special_tokens: bool = True, **kwargs) -> Dict:
        """
        Encode text into token IDs with additional information.

        Args:
            text: Text to encode
            add_special_tokens: Whether to add special tokens (ignored for tiktoken)
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary with 'input_ids' key containing token IDs
        """
        input_ids = self.tokenizer.encode(text)
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}

    def _encode_plus(self, text: str, **kwargs) -> Dict:
        """Internal method called by encode_plus."""
        return self.encode_plus(text, **kwargs)

    def _convert_token_to_id(self, token: str) -> int:
        return int(token)

    def _convert_id_to_token(self, index: int) -> str:
        return str(index)

    def get_vocab(self) -> Dict[str, int]:
        return dict(enumerate(range(self.vocab_size)))

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def save_vocabulary(self, *args) -> Tuple[str]:
        return ()

    def __len__(self) -> int:
        """Return the vocabulary size."""
        return self.vocab_size

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Class method to match HuggingFace's interface."""
        return cls()
