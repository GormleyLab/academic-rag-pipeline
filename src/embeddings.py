"""
Embedding generation module using OpenAI API.
"""

import logging
import time
from typing import Optional
from dataclasses import dataclass

from openai import OpenAI

from src.utils import setup_logger, clean_text_for_embedding


logger = setup_logger(__name__)


@dataclass
class EmbeddingResult:
    """Result of an embedding operation."""
    embedding: list[float]
    token_count: int
    model: str


class EmbeddingGenerator:
    """
    Generates embeddings using OpenAI's API.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-large",
        dimensions: int = 3072,
        batch_size: int = 100,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the embedding generator.

        Args:
            api_key: OpenAI API key
            model: Embedding model to use
            dimensions: Dimension of embeddings
            batch_size: Number of texts to embed in one API call
            max_retries: Maximum retry attempts for failed requests
            retry_delay: Initial delay between retries in seconds
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        logger.info(
            f"EmbeddingGenerator initialized with model={model}, "
            f"dimensions={dimensions}, batch_size={batch_size}"
        )

    def generate_embedding(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            EmbeddingResult containing the embedding vector

        Raises:
            Exception: If embedding generation fails after all retries
        """
        # Clean text
        cleaned_text = clean_text_for_embedding(text)

        if not cleaned_text:
            logger.warning("Empty text provided for embedding")
            # Return zero vector for empty text
            return EmbeddingResult(
                embedding=[0.0] * self.dimensions,
                token_count=0,
                model=self.model
            )

        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    input=cleaned_text,
                    model=self.model,
                    dimensions=self.dimensions
                )

                embedding = response.data[0].embedding
                token_count = response.usage.total_tokens

                return EmbeddingResult(
                    embedding=embedding,
                    token_count=token_count,
                    model=self.model
                )

            except Exception as e:
                logger.warning(
                    f"Embedding generation failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    delay = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"Failed to generate embedding after {self.max_retries} attempts")
                    raise

    def generate_embeddings_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of texts to embed

        Returns:
            List of EmbeddingResults in the same order as input texts

        Raises:
            Exception: If batch embedding fails
        """
        if not texts:
            return []

        results = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        logger.info(
            f"Processing {len(texts)} texts in {total_batches} batches "
            f"(batch_size={self.batch_size})"
        )

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1

            logger.info(f"Processing batch {batch_num}/{total_batches}")

            # Clean all texts in batch
            cleaned_batch = [clean_text_for_embedding(text) for text in batch]

            # Filter out empty texts but keep track of indices
            non_empty_indices = [
                idx for idx, text in enumerate(cleaned_batch) if text
            ]
            non_empty_texts = [cleaned_batch[idx] for idx in non_empty_indices]

            # Generate embeddings for non-empty texts
            if non_empty_texts:
                batch_results = self._generate_batch_with_retry(non_empty_texts)
            else:
                batch_results = []

            # Reconstruct results with zero vectors for empty texts
            result_iter = iter(batch_results)
            for idx, text in enumerate(cleaned_batch):
                if text:
                    results.append(next(result_iter))
                else:
                    # Zero vector for empty text
                    results.append(
                        EmbeddingResult(
                            embedding=[0.0] * self.dimensions,
                            token_count=0,
                            model=self.model
                        )
                    )

        logger.info(f"Successfully generated {len(results)} embeddings")
        return results

    def _generate_batch_with_retry(self, texts: list[str]) -> list[EmbeddingResult]:
        """
        Generate embeddings for a batch with retry logic.

        Args:
            texts: List of cleaned texts to embed

        Returns:
            List of EmbeddingResults

        Raises:
            Exception: If all retries fail
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.model,
                    dimensions=self.dimensions
                )

                # Extract embeddings and token counts
                results = []
                for data in response.data:
                    results.append(
                        EmbeddingResult(
                            embedding=data.embedding,
                            token_count=response.usage.total_tokens // len(texts),
                            model=self.model
                        )
                    )

                return results

            except Exception as e:
                logger.warning(
                    f"Batch embedding failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )

                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Failed to generate batch embeddings after {self.max_retries} attempts"
                    )
                    raise

    def estimate_cost(self, token_count: int) -> float:
        """
        Estimate the cost of embedding generation.

        Args:
            token_count: Number of tokens to embed

        Returns:
            Estimated cost in USD

        Note:
            Pricing for text-embedding-3-large: $0.00013 per 1K tokens
            (as of 2024 - check OpenAI pricing for updates)
        """
        # Price per 1K tokens for text-embedding-3-large
        price_per_1k = 0.00013

        cost = (token_count / 1000.0) * price_per_1k
        return cost

    def get_embedding_stats(self, results: list[EmbeddingResult]) -> dict:
        """
        Get statistics about embedding generation.

        Args:
            results: List of EmbeddingResults

        Returns:
            Dictionary with statistics
        """
        total_tokens = sum(r.token_count for r in results)
        estimated_cost = self.estimate_cost(total_tokens)

        return {
            "total_embeddings": len(results),
            "total_tokens": total_tokens,
            "estimated_cost_usd": estimated_cost,
            "model": self.model,
            "dimensions": self.dimensions,
            "avg_tokens_per_embedding": total_tokens / len(results) if results else 0
        }
