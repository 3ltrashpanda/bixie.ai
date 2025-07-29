import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import hashlib
import time
import logging
import joblib
import os
import sys

# rootpath = os.path.join(os.getcwd(), '..')
# sys.path.append(os.getcwd())
# os.environ.setdefault("PYTHONPATH","/home/trashpanda/repos/bixie.ai/")
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from bixie.vector_store.chroma_store import ChromaStore

# Constants
DEFAULT_MODEL_NAME = "huggingface/CodeBERTa-small-v1"
CLASSIFIER_PATH = Path("bixie.ai/fine_tuned_model.pkl")

logger = logging.getLogger("bixie.model_inference")

class SAFEEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def embed_binary_string(self, code_bytes: bytes) -> np.ndarray:
        """
        Embeds raw code (disassembled or bytecode) into a vector.
        """
        code_str = code_bytes.decode("utf-8", errors="ignore")
        tokens = self.tokenizer(code_str, return_tensors="pt", truncation=True, padding=True, max_length=512)

        with torch.no_grad():
            output = self.model(**tokens)
            embedding = output.last_hidden_state.mean(dim=1)  # Mean pooling

        return embedding.squeeze().numpy()

    def embed_file(self, filepath: Path) -> Optional[np.ndarray]:
        try:
            with open(filepath, "rb") as f:
                code_bytes = f.read()
            return self.embed_binary_string(code_bytes)
        except Exception as e:
            print(f"[ERROR] Failed to embed {filepath}: {e}")
            return None

def compare_embeddings(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Cosine similarity between two embeddings.
    """
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def load_classifier(classifier_path: Path = CLASSIFIER_PATH):
    if classifier_path.exists():
        try:
            clf = joblib.load(classifier_path)
            logger.info(f"Loaded trained classifier from {classifier_path}")
            return clf
        except Exception as e:
            logger.error(f"Failed to load classifier: {e}")
    else:
        logger.warning(f"No trained classifier found at {classifier_path}")
    return None

def run_model_inference(
    target_paths: List[Path],
    output_dir: Optional[Path] = None,
    save_vectors: bool = True,
    chroma_store: Optional["ChromaStore"] = None
) -> List[Dict]:
    """
    Run ML model inference on binaries or source files, generate embeddings,
    use trained classifier for prediction, and optionally persist them to Chroma vector store.

    Args:
        target_paths (List[Path]): List of files to analyze.
        output_dir (Optional[Path]): Directory for output (unused, for compatibility).
        save_vectors (bool): If True, save embeddings to Chroma.
        chroma_store (Optional[ChromaStore]): Chroma vector store instance.

    Returns:
        List[Dict]: List of inference results with metadata and predictions.
    """
    embedder = SAFEEmbedder()
    classifier = load_classifier()
    results = []

    for path in target_paths:
        if not path.is_file():
            logger.warning(f"Skipping non-file target: {path}")
            continue

        try:
            file_bytes = path.read_bytes()
            file_hash = hashlib.sha256(file_bytes).hexdigest()
        except Exception as e:
            logger.error(f"Failed to read {path}: {e}")
            continue

        try:
            embedding = embedder.embed_file(path)
        except Exception as e:
            logger.error(f"Embedding failed for {path}: {e}")
            continue

        if embedding is None:
            logger.warning(f"No embedding generated for {path}")
            continue

        metadata = {
            "file": str(path),
            "hash": file_hash,
            "timestamp": time.time(),
            "model": getattr(embedder, "model_name", embedder.__class__.__name__),
        }

        prediction = None
        prediction_label = None
        prediction_proba = None

        if classifier:
            try:
                prediction = classifier.predict([embedding])[0]
                if hasattr(classifier, "predict_proba"):
                    prediction_proba = classifier.predict_proba([embedding])[0]
                prediction_label = "vulnerable" if prediction == 1 else "good"
            except Exception as e:
                logger.error(f"Classifier prediction failed for {path}: {e}")

        # Save to Chroma vector store if requested
        if chroma_store and save_vectors:
            try:
                chroma_store.add_embedding(embedding, metadata)
            except Exception as e:
                logger.error(f"Failed to save embedding for {path}: {e}")

        results.append({
            "file": str(path),
            "embedding_id": file_hash,
            "metadata": metadata,
            "prediction": prediction_label,
            "prediction_raw": int(prediction) if prediction is not None else None,
            "prediction_proba": prediction_proba.tolist() if prediction_proba is not None else None,
        })

    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python model_inference.py <file1> <file2>")
        sys.exit(1)

    embedder = SAFEEmbedder()
    emb1 = embedder.embed_file(Path(sys.argv[1]))
    emb2 = embedder.embed_file(Path(sys.argv[2]))

    if emb1 is not None and emb2 is not None:
        sim = compare_embeddings(emb1, emb2)
        print(f"Cosine Similarity: {sim:.4f}")

