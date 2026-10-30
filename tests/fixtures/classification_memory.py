"""Real classification retrieval over private FAISS memory with explicit fake embeddings."""
import hashlib

import numpy as np


class ClassificationEmbeddingFixture:
    embedding_dim = 1024

    def embed_classification_prompt(self, prompt, classification_type='routing', side=None):
        data = hashlib.sha256(f'{classification_type}:{prompt}'.encode()).digest()
        values = np.tile(np.frombuffer(data, dtype=np.uint8), 32).astype(np.float32) + 1
        return values / np.linalg.norm(values)

    def embed_text(self, text, side=None):
        return self.embed_classification_prompt(text)


def install_classification_memory(isolation, root):
    from orchestration.repl_memory.episodic_store import EpisodicStore
    from orchestration.repl_memory.retriever import TwoPhaseRetriever
    from src.classifiers import classification_retriever

    store = EpisodicStore(db_path=root / 'classification-memory', embedding_dim=1024)
    retriever = TwoPhaseRetriever(store, ClassificationEmbeddingFixture())
    classifier = classification_retriever.ClassificationRetriever(retriever)
    isolation.setattr(classification_retriever, '_classification_retriever', classifier)
    return classifier, store
