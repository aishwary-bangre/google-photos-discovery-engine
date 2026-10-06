"""Embeddings: local sentence-transformers, with a TF-IDF+SVD fallback so the
pipeline never hard-fails if the model can't be downloaded."""
from __future__ import annotations

import numpy as np

from . import config as C

_model = None
_backend = None


def backend() -> str:
    _load()
    return _backend


def _load():
    global _model, _backend
    if _backend:
        return
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(C.EMBED_MODEL)
        _backend = f"sentence-transformers ({C.EMBED_MODEL})"
    except Exception as e:  # noqa: BLE001
        print(f"[embed] sentence-transformers unavailable ({e}); using TF-IDF+SVD fallback")
        _backend = "tfidf-svd"


def encode(texts: list[str], fit_corpus: list[str] | None = None) -> np.ndarray:
    """Return L2-normalised vectors. For the TF-IDF fallback, `fit_corpus` defines
    the vocabulary so anchors and documents share one space."""
    _load()
    if _model is not None:
        v = _model.encode(texts, batch_size=64, show_progress_bar=len(texts) > 500,
                          normalize_embeddings=True)
        return np.asarray(v, dtype=np.float32)
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    corpus = fit_corpus or texts
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english", sublinear_tf=True)
    vec.fit(corpus)
    X = vec.transform(texts)
    k = max(2, min(128, X.shape[1] - 1, len(corpus) - 1))
    svd = TruncatedSVD(n_components=k, random_state=42).fit(vec.transform(corpus))
    return normalize(svd.transform(X)).astype(np.float32)
