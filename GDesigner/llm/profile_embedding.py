import os
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

_LOCAL_MODEL_PATH = "/data/lyz/models/all-MiniLM-L6-v2"

_model = None


def _get_model():
    global _model
    if _model is None:
        model_path = os.getenv("MINILM_MODEL_PATH", _LOCAL_MODEL_PATH)
        if not os.path.isdir(model_path):
            model_path = "sentence-transformers/all-MiniLM-L6-v2"
        _model = SentenceTransformer(model_path)
    return _model


def get_sentence_embedding(sentence):
    embeddings = _get_model().encode(sentence)
    return embeddings


@lru_cache(maxsize=256)
def _get_sentence_embeddings_cached(sentences):
    return _get_model().encode(
        list(sentences),
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def get_sentence_embeddings(sentences):
    """Encode an ordered text batch once and return a caller-owned array."""
    sentence_tuple = tuple(str(sentence) for sentence in sentences)
    return np.array(
        _get_sentence_embeddings_cached(sentence_tuple),
        copy=True,
    )
