from typing import List, Tuple

import numpy as np

from config.embed_config import EmbeddingConnector
from enums.embed_type import EmbedType


class Embedding:
    """Utility class to instantiate various embedding providers from configuration.

    Supported providers are OpenAI, HuggingFace, and Ollama.
    """

    def __init__(
        self,
        model_name: str,
        embed_type: EmbedType = EmbedType.OPEN_AI,
        api_key: str = None,
        endpoint: str = None,
    ):
        self.embed = EmbeddingConnector(
            model_name=model_name,
            embed_type=embed_type,
            api_key=api_key,
            endpoint=endpoint,
        )

    def get_embedding(self, texts: List[str], batch_size: int) -> Tuple[np.ndarray, int]:
        """
        Return (embeddings array, total tokens used).
        Tokens are non-zero only for OpenAI; Ollama/HuggingFace return 0.
        """
        if batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")

        if self.embed.embed_type == EmbedType.OPEN_AI:
            return self._get_openai_embedding_with_usage(texts, batch_size)
        if self.embed.embed_type == EmbedType.HUGGINGFACE:
            arr = self.embed().encode(
                texts, batch_size=batch_size, convert_to_numpy=True
            )
            return arr, 0
        # Ollama
        vectors: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch_text = texts[i : i + batch_size]
            batch_vectors = self.embed().embed_documents(batch_text)
            vectors.extend(batch_vectors)
        arr = np.asarray(vectors, dtype=np.float32)
        return arr, 0

    def _get_openai_embedding_with_usage(
        self, texts: List[str], batch_size: int
    ) -> Tuple[np.ndarray, int]:
        """Use OpenAI client directly to capture token usage."""
        from openai import OpenAI

        client = OpenAI(api_key=self.embed.api_key)
        vectors: List[List[float]] = []
        total_tokens = 0
        for i in range(0, len(texts), batch_size):
            batch_text = texts[i : i + batch_size]
            resp = client.embeddings.create(model=self.embed.model, input=batch_text)
            for d in resp.data:
                vectors.append(d.embedding)
            if getattr(resp, "usage", None) is not None:
                total_tokens += getattr(resp.usage, "total_tokens", 0) or 0
        arr = np.asarray(vectors, dtype=np.float32)
        return arr, total_tokens