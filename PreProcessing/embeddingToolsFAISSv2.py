import json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os


class EmbeddingToolFAISS:
    def __init__(
        self,
        output_dir: Path,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_backend: str = "hnsw",  # "faiss" (flat) or "hnsw"
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 128,
        title_weight: int = 2,
        store_retrieval_doc: bool = True,
    ):
        self.model = SentenceTransformer(model_name)
        self.index_dir = output_dir / "index"
        os.makedirs(self.index_dir, exist_ok=True)

        # Index backend settings
        self.index_backend = index_backend.lower()
        self.hnsw_m = int(hnsw_m)
        self.hnsw_ef_construction = int(hnsw_ef_construction)
        self.hnsw_ef_search = int(hnsw_ef_search)

        # Retrieval doc settings
        self.title_weight = int(title_weight)
        self.store_retrieval_doc = bool(store_retrieval_doc)

        self.index = None
        self.metadata = []
        self.embeddings: np.ndarray | None = None

    def _make_retrieval_doc(self, r: dict) -> str:
        title = (r.get("chunk_title") or "").strip()
        text = (r.get("chunk_text") or "").strip()

        if title:
            w = max(1, self.title_weight)
            title_w = " ".join([title] * w)
            return f"{title_w}. {text}".strip()

        return text

    # --- Build index ---
    def build_index(self, chunks, verbose: bool = True, save: bool = True):
        """
        Loads chunks, computes embeddings, builds index.
        Saves:
          - index/faiss.index
          - index/docs.json
        """
        if not chunks:
            raise ValueError("No chunks found. Please run split_into_chunks_json() first.")

        print(f"Encoding {len(chunks)} chunks...")

        texts = []
        for r in chunks:
            retrieval_doc = self._make_retrieval_doc(r)
            texts.append(retrieval_doc)
            if self.store_retrieval_doc:
                r["_retrieval_doc"] = retrieval_doc

        embs = self.embed_texts(texts, self.model)

        # Initialize index backend
        dim = embs.shape[1]
        if self.index_backend == "faiss":
            self.index = faiss.IndexFlatIP(dim)
        elif self.index_backend == "hnsw":
            self.index = faiss.IndexHNSWFlat(dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = self.hnsw_ef_construction
            self.index.hnsw.efSearch = self.hnsw_ef_search
        else:
            raise ValueError("index_backend must be either 'faiss' or 'hnsw'")

        self.index.add(embs)

        # Keep embeddings in memory for MMR
        self.embeddings = embs

        # Persist index + metadata
        faiss.write_index(self.index, str(self.index_dir / "faiss.index"))
        if save:
            with open(self.index_dir / "docs.json", "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)

        if verbose:
            print(f"✅ Index built with {len(chunks)} chunks.")
            print(f"   • Backend: {self.index_backend.upper()}")
            print(f"   • Vectors: {embs.shape}")
            print(f"   • Saved index: {self.index_dir / 'faiss.index'}")
            print(f"   • Saved docs:  {self.index_dir / 'docs.json'}")
            if self.store_retrieval_doc:
                print("   • Added field: _retrieval_doc (title-weighted title + text)")

        return self.index

    # --- Generate embeddings ---
    def embed_texts(self, texts, model: SentenceTransformer, batch_size: int = 64):
        """
        Encodes a list of strings into L2-normalized float32 embeddings.
        Normalization enables cosine similarity via inner product in FAISS.
        """
        embs = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        if embs.dtype != np.float32:
            embs = embs.astype("float32")
        return embs

    # --- Load existing FAISS index + metadata ---
    def load_index(self):
        """
        Loads FAISS index + metadata from disk.
        """
        index_path = self.index_dir / "faiss.index"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}. Please build the index first.")
        self.index = faiss.read_index(str(index_path))

        docs_path = self.index_dir / "docs.json"
        if not docs_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {docs_path}. Please build the index first.")
        with open(docs_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.embeddings = None

        print(f"✅ Loaded FAISS index with {self.index.ntotal} vectors from {index_path}")
        print(f"   • Loaded metadata for {len(self.metadata)} documents from {docs_path}")

        return self.index, self.metadata
