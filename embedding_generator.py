import os
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_DIM = 768

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        cache_path: str = "matrix_cache.npy",
        device: str = None,
    ):
        self._model_name: str = model_name
        self._cache_path: str = cache_path

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device: str = device

        self._model = SentenceTransformer(model_name, device=device)
        self._tokenizer = self._model.tokenizer
        self._transformer_model = self._model._first_module().auto_model

        self.embeddings_matrix: np.ndarray = None

    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
        device: str = None,
        force_recompute: bool = False,
    ) -> np.ndarray:
        effective_device = device if device is not None else self._device

        # якщо кеш валідний — читаємо з диска
        if not force_recompute and os.path.exists(self._cache_path):
            try:
                cached = np.load(self._cache_path, mmap_mode="r")
                if cached.shape[0] == len(texts) and cached.shape[1] == self.EMBEDDING_DIM:
                    self.embeddings_matrix = np.array(cached, dtype=np.float32)
                    return self.embeddings_matrix
            except (OSError, ValueError):
                pass

        try:
            self.embeddings_matrix = self._model.encode(
                texts,
                batch_size=batch_size,
                device=effective_device,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=False,
            ).astype(np.float32)
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(
                "Переповнення GPU. Зменшіть batch_size або використайте device='cpu'."
            ) from exc
        except ValueError as exc:
            raise ValueError(f"Помилка вхідних даних для інференсу: {exc}") from exc

        np.save(self._cache_path, self.embeddings_matrix)
        return self.embeddings_matrix

    def encode_query(self, query: str) -> np.ndarray:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Запит не може бути порожнім")

        return self._model.encode(
            [query],
            batch_size=1,
            device=self._device,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)[0]

    def memory_footprint_mb(self) -> float:
        if self.embeddings_matrix is None:
            return 0.0
        return self.embeddings_matrix.nbytes / (1024 * 1024)
