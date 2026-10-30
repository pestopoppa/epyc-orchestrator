"""Explicit deterministic embedding input for model-free graph/REPL producer tests."""
from functools import wraps
import hashlib

import numpy as np


class ResearchEmbeddingFixture:
    embedding_dim = 1024

    def embed_sync(self, text, side=None):
        bits = np.unpackbits(np.frombuffer(hashlib.sha256(text.encode()).digest(), dtype=np.uint8))
        return (np.tile(bits, 4).astype(np.float32) * 2 - 1) / 32

    def close_sync(self):
        pass


def install_research_embeddings(isolation):
    from src.research_context import ResearchContext

    original = ResearchContext.__init__

    @wraps(original)
    def owned_init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if self.use_semantic and self.embedder is None:
            self.embedder = ResearchEmbeddingFixture()

    isolation.setattr(ResearchContext, '__init__', owned_init)
