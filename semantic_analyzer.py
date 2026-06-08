from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class SemanticAnalyzer:
    """Косинусна подібність + відбір top-k + нелінійне масштабування ваг (w = score^k)."""

    def __init__(
        self,
        similarity_threshold: float = 0.3,
        exponent_k: int = 3,
        positive_only: bool = True,
    ):
        self._similarity_threshold: np.float32 = np.float32(similarity_threshold)
        self._exponent_k: int = exponent_k
        self._positive_only: bool = positive_only
        self._similarity_scores: np.ndarray = None
        self.adjacency_matrix: csr_matrix = None

    def analyze_similarity(
        self,
        query_vector: np.ndarray,
        global_matrix: np.ndarray,
        top_k: int = 100,
    ) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
        try:
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)

            if query_vector.shape[1] != global_matrix.shape[1]:
                raise ValueError(
                    f"Розмірність запиту {query_vector.shape[1]} "
                    f"!= розмірності корпусу {global_matrix.shape[1]}"
                )

            scores = cosine_similarity(query_vector, global_matrix)[0].astype(np.float32)
            self._similarity_scores = scores

            effective_k = min(top_k, len(scores))
            top_indices = np.argpartition(scores, -effective_k)[-effective_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            top_scores = scores[top_indices]

            local_vectors = global_matrix[top_indices]
            pairwise = cosine_similarity(local_vectors).astype(np.float32)

            if self._positive_only:
                pairwise = np.where(pairwise > 0, pairwise, 0.0)
            pairwise = np.power(pairwise, self._exponent_k)
            pairwise[pairwise < self._similarity_threshold] = 0.0
            np.fill_diagonal(pairwise, 0.0)

            self.adjacency_matrix = csr_matrix(pairwise)
            return self.adjacency_matrix, top_indices, top_scores

        except ValueError as exc:
            raise ValueError(f"Невідповідність розмірностей: {exc}") from exc
        except AttributeError as exc:
            raise AttributeError(f"Помилка атрибутів: {exc}") from exc

    def get_raw_scores(self) -> np.ndarray:
        return self._similarity_scores
