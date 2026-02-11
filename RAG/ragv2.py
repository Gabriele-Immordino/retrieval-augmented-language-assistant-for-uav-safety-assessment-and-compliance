"""
RAG retrieval stack with field aware logic.

Field usage
1. chunk_title
   Used in first stage retrieval via chunk["_retrieval_doc"] which should already be title weighted plus text.

2. chunk_summary
   Used only after first stage retrieval as an extra semantic signal to refine ranking.

3. chunk_keywords
   Used as a light lexical boost, never embedded as the main document.
"""

from RAG.ragAbs import RAGSystem
from PreProcessing.embeddingToolsFAISSv2 import EmbeddingToolFAISS

import numpy as np
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModel = None


def make_retrieval_doc(r: dict) -> str:
    """
    Document used for first stage retrieval.

    Preferred is r["_retrieval_doc"] created at indexing time.
    Fallback builds title plus text.
    """
    doc = (r.get("_retrieval_doc") or "").strip()
    if doc:
        return doc

    title = (r.get("chunk_title") or "").strip()
    text = (r.get("chunk_text") or "").strip()
    if title:
        return f"{title}. {text}".strip()
    return text


def make_rerank_doc(r: dict) -> str:
    """
    Document used for reranking.

    Includes title, summary, and text when available.
    """
    title = (r.get("chunk_title") or "").strip()
    summary = (r.get("chunk_summary") or "").strip()
    text = (r.get("chunk_text") or "").strip()

    if title and summary:
        return f"{title}. {summary}\n{text}".strip()
    if title:
        return f"{title}\n{text}".strip()
    if summary:
        return f"{summary}\n{text}".strip()
    return text


def keyword_boost(query: str, keywords: list[str] | None) -> float:
    """
    Lightweight lexical boost from chunk keywords.

    Returns a small score based on keyword matches in the query.
    Keep this small so it never dominates dense or reranker scores.
    """
    if not keywords:
        return 0.0
    q = query.lower()
    hits = 0
    for k in keywords:
        kk = (k or "").lower().strip()
        if kk and kk in q:
            hits += 1
    return float(hits)


class DenseRetriever:
    """
    Dense semantic retrieval using FAISS plus MMR.

    Assumes index vectors are normalized.
    Query embeddings are normalized to make inner product equal cosine similarity.
    """

    def __init__(self, embedding_tool: EmbeddingToolFAISS, chunks: list[dict] | None = None):
        if embedding_tool.index is None:
            raise ValueError("EmbeddingTool.index is None. Build or load the FAISS index first.")
        if chunks is None:
            raise ValueError("chunks is None. Provide the same chunk list used to build docs.json.")

        self.embedding_tool = embedding_tool
        self.index = embedding_tool.index
        self.model = embedding_tool.model
        self.chunks = chunks

        self.embeddings = embedding_tool.embeddings

    def _mmr_select(self, qvec: np.ndarray, doc_vecs: np.ndarray, k: int, lam: float = 0.6) -> list[int]:
        """
        Max Marginal Relevance, returns indices into doc_vecs.
        lam controls the tradeoff between relevance and diversity.
        """
        n = doc_vecs.shape[0]
        if n == 0 or k <= 0:
            return []
        
        # Relevance to the query, cosine since vectors are normalized
        rel = doc_vecs @ qvec
        selected: list[int] = []
        remaining = list(range(n))

        # Pick the most relevant first
        first = int(np.argmax(rel))
        selected.append(first)
        remaining.remove(first)

        while remaining and len(selected) < k:
            # For each remaining, compute its max similarity to any selected
            red = np.array([float(np.max(doc_vecs[r] @ doc_vecs[selected].T)) for r in remaining], dtype=np.float32)
            mmr_scores = lam * rel[remaining] - (1.0 - lam) * red
            best_idx = int(np.argmax(mmr_scores))
            pick = remaining[best_idx]
            selected.append(pick)
            remaining.pop(best_idx)

        return selected

    def search(self, query: str, top_k: int = 50, mmr_lambda: float = 0.6, initial_k: int | None = None) -> list[dict]:
        """
        Retrieve with FAISS, then apply MMR.
        Uses raw inner-product similarities (no vector normalization or thresholds here).
        """
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")[0]

        pool_k = initial_k or max(50, 5 * top_k)
        scores, idxs = self.index.search(q_emb[None, :], pool_k)
        idxs = idxs[0]

        # Filter out invalid ids
        cand_ids = [int(i) for i in idxs if 0 <= int(i) < len(self.chunks)]
        if not cand_ids:
            return []

        cand_ids_np = np.array(cand_ids, dtype=int)

        # Get document vectors for MMR
        if isinstance(self.embeddings, np.ndarray):
            doc_vecs = self.embeddings[cand_ids_np]
        else:
            # Fallback: reconstruct from FAISS
            vecs = []
            keep = []
            for i in cand_ids_np:
                try:
                    vecs.append(self.index.reconstruct(int(i)))
                    keep.append(int(i))
                except Exception:
                    pass
            if not keep:
                return []
            cand_ids_np = np.array(keep, dtype=int)
            doc_vecs = np.vstack(vecs).astype("float32")

        # Apply MMR on the candidate pool
        mmr_order_local = self._mmr_select(q_emb, doc_vecs, k=top_k, lam=mmr_lambda)
        if not mmr_order_local:
            return []

        sel_ids = cand_ids_np[mmr_order_local]
        sel_vecs = doc_vecs[mmr_order_local]
        # Compute raw inner-product similarity for selected docs
        cosines = sel_vecs @ q_emb

        results: list[dict] = []
        for rank, (doc_id, cos) in enumerate(zip(sel_ids, cosines), start=1):
            rec = dict(self.chunks[int(doc_id)])
            rec["_rank"] = rank
            rec["_score"] = float(cos)
            rec["_id"] = int(doc_id)
            results.append(rec)

        # Ensure results are ordered by descending score; fix ranks if needed
        results.sort(key=lambda r: r.get("_score", 0.0), reverse=True)
        for i, r in enumerate(results, start=1):
            r["_rank"] = i

        return results


class BM25Retriever:
    """
    BM25 lexical retriever over the retrieval document.
    This means BM25 also benefits from chunk_title via _retrieval_doc.
    """

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.corpus = [make_retrieval_doc(c) for c in chunks]
        self.tokenized_corpus = [self._tokenize(text) for text in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        text = text.lower()
        return re.findall(r"\w+", text)

    def search(self, query: str, top_k: int = 50) -> list[dict]:
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)

        if len(scores) == 0:
            return []

        scores = np.asarray(scores, dtype="float32")
        # Sort descending
        top_idx = np.argsort(scores)[::-1][:top_k]

        results: list[dict] = []
        for rank, doc_id in enumerate(top_idx, start=1):
            rec = dict(self.chunks[int(doc_id)])
            rec["_rank"] = rank
            rec["_score"] = float(scores[int(doc_id)])
            rec["_id"] = int(doc_id)
            results.append(rec)

        return results


class RAG(RAGSystem):
    """
    Orchestrator for dense, bm25, or hybrid retrieval with field aware post scoring.

    Additional post scoring signals:
    - summary semantic score computed only on the candidate set
    - keyword boost computed as a light lexical bonus
    """

    def __init__(
        self,
        embedding_tool: EmbeddingToolFAISS,
        chunks: list[dict] | None = None,
        default_mode: str = "hybrid",
        cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        reranker: str = "colbert",
        use_summary_score: bool = True,
        use_keyword_boost: bool = True,
        summary_weight: float = 0.5,
        keyword_weight: float = 0.5,
    ):
        """
        Initialize the BaseRAG retriever.

        Parameters:
          embedding_tool (EmbeddingTool): EmbeddingTool with built or loaded FAISS index.
          default_mode (str): One of "dense", "bm25", or "hybrid".
        """
        super().__init__(embedding_tool)

        if chunks is None:
            raise ValueError("chunks is None. Provide the same chunk list used to build docs.json.")
        if default_mode not in ("dense", "bm25", "hybrid"):
            raise ValueError("default_mode must be one of 'dense', 'bm25', or 'hybrid'.")
        if reranker.lower() not in ("ce", "colbert", "none"):
            raise ValueError("reranker must be one of 'ce', 'colbert', or 'none'.")

        self.chunks = chunks
        self.index = self.embedding_tool.index

        self.dense_retriever = DenseRetriever(embedding_tool=self.embedding_tool, chunks=self.chunks)

        self.bm25_retriever: BM25Retriever | None = None
        if default_mode in ("bm25", "hybrid"):
            self.bm25_retriever = BM25Retriever(chunks=self.chunks)

        self.default_mode = default_mode

        # Cross-encoder
        self.cross_encoder_model_name = cross_encoder_model_name
        self.cross_encoder: CrossEncoder | None = None

        self.reranker = reranker.lower()

        # Reranking strategy: 'ce' | 'colbert' | 'none'
        self._colbert_tok = None
        self._colbert_model = None
        self._colbert_device = None

        self.use_summary_score = bool(use_summary_score)
        self.use_keyword_boost = bool(use_keyword_boost)
        self.summary_weight = float(summary_weight)
        self.keyword_weight = float(keyword_weight)

    @staticmethod
    def _rrf_fusion(lists: dict[str, list[dict]], rrf_k: int = 60, max_results: int | None = None) -> list[dict]:
        """
        Reciprocal Rank Fusion over multiple ranked lists.

        lists: {retriever_name: [result_dicts_with__id_and__rank]}
        RRF(d) = Σ_i 1 / (k + rank_i(d))
        """
        scores: dict[int, float] = {}
        for _, res_list in lists.items():
            for rec in res_list:
                doc_id = int(rec["_id"])
                rank = int(rec["_rank"])
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)

        ordered_ids = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
        if max_results is not None:
            ordered_ids = ordered_ids[:max_results]

        fused = []
        for pos, doc_id in enumerate(ordered_ids, start=1):
            fused.append({"_id": int(doc_id), "_rank": pos, "_score": float(scores[doc_id])})
        return fused

    def _apply_summary_score(self, query: str, results: list[dict]) -> None:
        """
        Adds a semantic score based on chunk_summary to each result in place.
        Uses the same embedding model as dense retrieval.
        """
        if not results:
            return

        model = self.embedding_tool.model
        q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")[0]

        summaries = [(r.get("chunk_summary") or "").strip() for r in results]
        if not any(summaries):
            for r in results:
                r["_summary_score"] = 0.0
            return

        s_emb = model.encode(summaries, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        sim = s_emb @ q
        for r, sc in zip(results, sim):
            r["_summary_score"] = float(sc)

    def _apply_keyword_boost(self, query: str, results: list[dict]) -> None:
        """
        Adds a small keyword match score to each result in place.
        """
        for r in results:
            r["_kw_boost"] = keyword_boost(query, r.get("chunk_keywords"))

    def _combine_post_scores(self, results: list[dict]) -> list[dict]:
        """
        Combines base score with optional summary score and keyword boost.

        The base score comes from the current retrieval or reranking stage.
        """
        if not results:
            return results

        for r in results:
            base = float(r["_score"])*100  # Scale base to be comparable
            if self.use_summary_score:
                base = (1.0 - self.summary_weight) * base + self.summary_weight * float(r["_summary_score"])
            if self.use_keyword_boost:
                base = base + self.keyword_weight * float(r["_kw_boost"])
            r["_score"] = float(base)

        results.sort(key=lambda r: r.get("_score", 0.0), reverse=True)
        for i, r in enumerate(results, start=1):
            r["_rank"] = i
        return results

    def search(
        self,
        query: str,
        mode: str | None = None,
        top_k: int = 50,
        mmr_lambda: float = 0.6,
        initial_k: int | None = None,
        bm25_top_k: int | None = None,
        dense_top_k: int | None = None,
        rrf_k: int = 60,
        ce_keep_k: int | None = None,
    ) -> list[dict]:
        """
        Unified search entrypoint with optional reranking applied after retrieval.

        mode:
          - "dense": FAISS + MMR
          - "bm25":  BM25 only
          - "hybrid": BM25 + dense fused with RRF

        top_k: number of final results to return.
        ce_keep_k: candidate pool size kept for reranking in hybrid (defaults to top_k).
        """
        mode = (mode or self.default_mode).lower()
        results: list[dict] = []

        if mode == "dense":
            results = self.dense_retriever.search(query=query, top_k=top_k, mmr_lambda=mmr_lambda, initial_k=initial_k)

        elif mode == "bm25":
            if self.bm25_retriever is None:
                self.bm25_retriever = BM25Retriever(chunks=self.chunks)
            results = self.bm25_retriever.search(query=query, top_k=top_k)

        elif mode == "hybrid":
            if self.bm25_retriever is None:
                self.bm25_retriever = BM25Retriever(chunks=self.chunks)

            n_bm25 = bm25_top_k or top_k
            n_dense = dense_top_k or top_k
            n_keep = ce_keep_k or top_k

            dense_results = self.dense_retriever.search(query=query, top_k=n_dense, mmr_lambda=mmr_lambda, initial_k=initial_k)
            bm25_results = self.bm25_retriever.search(query=query, top_k=n_bm25)

            fused_meta = self._rrf_fusion({"dense": dense_results, "bm25": bm25_results}, rrf_k=rrf_k, max_results=n_keep)

            results = []
            for meta in fused_meta:
                doc_id = int(meta["_id"])
                chunk = dict(self.chunks[doc_id])
                chunk["_id"] = doc_id
                chunk["_rank"] = int(meta["_rank"])
                chunk["_score"] = float(meta["_score"])
                results.append(chunk)
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}. Use 'dense', 'bm25', or 'hybrid'.")

        # Field aware post scoring before heavy rerankers
        if results:
            if self.use_summary_score:
                self._apply_summary_score(query, results)
            if self.use_keyword_boost:
                self._apply_keyword_boost(query, results)
            results = self._combine_post_scores(results)

        # Optional heavy reranking after the lightweight field logic
        if results and self.reranker != "none":
            if self.reranker == "ce":
                results = self._rerank_with_cross_encoder(query, results)
            elif self.reranker == "colbert":
                results = self._rerank_with_colbert(query, results)

            # # Reapply lightweight field logic after reranking to keep keywords and summary in play
            # if results:
            #     if self.use_summary_score:
            #         self._apply_summary_score(query, results)
            #     if self.use_keyword_boost:
            #         self._apply_keyword_boost(query, results)
            #     results = self._combine_post_scores(results)

        if results:
            results = self._filter_results_by_elbow(results, min_keep=1, max_keep=top_k)    
            if len(results) > top_k:
                results = results[:top_k]
            for i, r in enumerate(results, start=1):
                r["_rank"] = i

        return results

    def _rerank_with_cross_encoder(self, query: str, results: list[dict]) -> list[dict]:
        """
        Cross encoder reranking using title, summary, and text in the doc field.
        """
        if not results:
            return results
        if self.cross_encoder is None:
            self.cross_encoder = CrossEncoder(self.cross_encoder_model_name)

        pairs = [(query, make_rerank_doc(r)) for r in results]
        ce_scores = self.cross_encoder.predict(pairs, batch_size=min(32, len(pairs)), show_progress_bar=False)

        for r, s in zip(results, ce_scores):
            r["_score"] = float(s)
        results.sort(key=lambda r: r.get("_score", 0.0), reverse=True)
        for i, r in enumerate(results, start=1):
            r["_rank"] = i
        return results

    @staticmethod
    def encode_tokens(texts: list[str], max_len: int, tok, model, device):
        """
        Encode token embeddings and valid token mask for late interaction scoring.
        """
        inputs = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
            X = out.last_hidden_state
            attn = inputs.get("attention_mask").bool()
            ids = inputs.get("input_ids")

            mask = attn.clone()
            for sid in [tok.cls_token_id, tok.sep_token_id, tok.pad_token_id]:
                if sid is not None:
                    mask &= (ids != sid)

            X = torch.nn.functional.normalize(X, p=2, dim=-1)
            return X, mask

    def _rerank_with_colbert(self, query: str, results: list[dict]) -> list[dict]:
        """
        ColBERT style reranking using title, summary, and text.

        If torch or transformers are unavailable, returns unchanged results.
        """
        if not results:
            return results
        if torch is None or AutoTokenizer is None or AutoModel is None:
            return results

        if self._colbert_tok is None or self._colbert_model is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tok = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")
            model = AutoModel.from_pretrained("colbert-ir/colbertv2.0")
            model.eval()
            model.to(device)
            if device.type == "cuda":
                try:
                    model.half()
                except Exception:
                    pass
            self._colbert_tok = tok
            self._colbert_model = model
            self._colbert_device = device

        tok = self._colbert_tok
        model = self._colbert_model
        device = self._colbert_device

        q_emb, q_mask = RAG.encode_tokens([query], max_len=64, tok=tok, model=model, device=device)
        q_emb = q_emb[0][q_mask[0]]
        if q_emb.numel() == 0:
            return results

        texts = [make_rerank_doc(r) for r in results]
        d_emb, d_mask = RAG.encode_tokens(texts, max_len=180, tok=tok, model=model, device=device)

        if d_emb.size(1) == 0:
            for r in results:
                r["_score"] = 0.0
            return results

        sim = torch.einsum("bth,qh->btq", d_emb, q_emb)
        sim = sim.masked_fill((~d_mask).unsqueeze(-1), float("-inf"))
        max_per_q = sim.max(dim=1).values
        max_per_q[max_per_q == float("-inf")] = 0.0
        scores_all = [float(s) for s in max_per_q.sum(dim=1).detach().cpu().tolist()]

        for r, s in zip(results, scores_all):
            r["_score"] = float(s)

        results.sort(key=lambda r: r.get("_score", 0.0), reverse=True)
        for i, r in enumerate(results, start=1):
            r["_rank"] = i
        return results

    def _filter_results_by_elbow(
        self,
        results: list[dict],
        min_keep: int = 1,
        drop_threshold: float = 0.8,
        max_keep: int | None = None,
        score_key: str = "_score",
    ) -> list[dict]:
        """
        Elbow cutoff on raw score differences.

        Steps
        1. Sort results by score_key descending.
        2. Compute consecutive drops: drop_i = score[i] - score[i+1]
        3. Keep prefix until first drop > drop_threshold.
        Always keep at least min_keep, and never exceed max_keep.

        This assumes that scores are already on a meaningful and
        roughly comparable scale within the result list.
        """
        if not results:
            return results

        results.sort(key=lambda r: float(r.get(score_key, 0.0)), reverse=True)

        n = len(results)
        if n <= min_keep:
            for i, r in enumerate(results, start=1):
                r["_rank"] = i
            return results

        scores = [float(r.get(score_key, 0.0)) for r in results]

        keep = min_keep
        limit = n if max_keep is None else min(n, int(max_keep))

        for i in range(min_keep - 1, limit - 1):
            drop = scores[i] - scores[i + 1]
            if drop > drop_threshold:
                keep = i + 1
                break
            keep = i + 2

        selected = results[:keep]
        for i, r in enumerate(selected, start=1):
            r["_rank"] = i
        return selected
