import os

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
