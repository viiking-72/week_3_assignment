#!/usr/bin/env python3
"""
Retrieval Pipeline for SIGGRAPH 2025 Papers.

Implements hybrid search:
1. Semantic search (embeddings + Qdrant Cloud)
2. Keyword search (BM25 - runs locally)
3. Reranking (Cohere API - optional)

The embedding provider is pluggable so the same code runs locally and on a
small deployment host. Set EMBEDDING_PROVIDER in .env:

    local        - run BAAI/bge-large-en-v1.5 on this machine via fastembed.
                   Free, no API key, ~42 ms/query, but needs ~2.4 GB RAM.
    huggingface  - call Hugging Face Inference for the same model.
                   Free tier, no credit card, tiny RAM. Use this on Render.
    openrouter   - call OpenRouter (original assignment path, needs credits).

All three MUST serve BAAI/bge-large-en-v1.5. The vectors already stored in
Qdrant came from that model; any other model produces a different vector
space and silently returns nonsense.

Usage:
    from retrieval_pipeline import RetrievalPipeline

    pipeline = RetrievalPipeline()
    results = pipeline.retrieve("3D Gaussian Splatting", top_k=5)
"""

import json
import os
import re
import requests
import numpy as np
from typing import Optional, List
from dataclasses import dataclass
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from dotenv import load_dotenv
load_dotenv()

# Must match the collection name used in upload_from_npz.py / upload_to_qdrant.py
COLLECTION_NAME = os.getenv("COLLECTION_NAME") or "siggraph2025_papers"

# Dimension of BAAI/bge-large-en-v1.5
VECTOR_SIZE = 1024


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean from the environment ("true"/"1"/"yes" -> True)."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _resolve_path(path: str) -> str:
    """Resolve a path from the cwd, falling back to this file's directory."""
    if os.path.exists(path):
        return path
    alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.basename(path))
    return alt if os.path.exists(alt) else path


@dataclass
class RetrievalResult:
    """
    Represents a single search result.
    The api_server.py expects these exact fields - do not change!
    """
    chunk_id: str
    paper_id: str
    title: str
    authors: str
    text: str
    score: float
    chunk_type: str = ""
    chunk_section: str = ""
    pdf_url: Optional[str] = None
    github_link: Optional[str] = None
    video_link: Optional[str] = None
    acm_url: Optional[str] = None
    abstract_url: Optional[str] = None


@dataclass
class RetrievalPipelineConfig:
    """Configuration for the retrieval pipeline."""
    qdrant_url: str
    qdrant_api_key: str
    openrouter_api_key: str
    embedding_model: str = "baai/bge-large-en-v1.5"
    chunks_path: str = "./chunks.json"
    semantic_weight: float = 0.7
    bm25_weight: float = 0.3
    use_reranker: bool = True
    cohere_api_key: Optional[str] = None
    # Cohere rerank model (v2 API).
    rerank_model: str = "rerank-v3.5"
    # Which service embeds the query: "local" | "huggingface" | "openrouter"
    embedding_provider: str = "local"
    huggingface_api_key: Optional[str] = None
    # When True, the chunk corpus is dropped after building the BM25 index and
    # payloads are fetched from Qdrant instead. This is what makes the backend
    # fit on a 512 MB host such as Render's free tier.
    low_memory: bool = False
    # Prebuilt compact BM25 index. Used only in low-memory mode, when present.
    bm25_index_path: str = "./bm25_index.npz"


# =============================================================================
# EMBEDDERS - one per provider, all returning BAAI/bge-large-en-v1.5 vectors
# =============================================================================
def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """BGE vectors are compared with cosine; normalising keeps scores comparable."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class OpenRouterEmbedder:
    """
    Generate embeddings using OpenRouter API.
    Used to embed user queries for semantic search.
    """

    def __init__(self, api_key: str, model: str = "baai/bge-large-en-v1.5"):
        """
        Initialize the embedder.

        Args:
            api_key: OpenRouter API key
            model: Embedding model to use
        """
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"

    def embed_query(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector as numpy array
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": text,
        }

        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=headers,
            json=payload,
            timeout=60,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"OpenRouter embeddings request failed "
                f"({response.status_code}): {response.text[:300]}"
            )

        response_data = response.json()
        embedding = response_data["data"][0]["embedding"]
        return _l2_normalize(np.array(embedding, dtype=np.float32))


class HuggingFaceEmbedder:
    """
    Generate embeddings using the Hugging Face Inference API.

    This is the deployment-friendly option: the model runs on Hugging Face's
    servers, so the backend itself stays small enough for a 512 MB host.
    The free tier needs only an account, no credit card.
    Get a token at https://huggingface.co/settings/tokens
    """

    # The router is the current endpoint; the legacy host is kept as a fallback
    # because Hugging Face has moved this URL more than once.
    ENDPOINTS = (
        "https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction",
        "https://api-inference.huggingface.co/models/{model}",
    )

    def __init__(self, api_key: str, model: str = "BAAI/bge-large-en-v1.5"):
        self.api_key = api_key
        # Hugging Face model ids are case sensitive, unlike OpenRouter's.
        self.model = "BAAI/bge-large-en-v1.5" if model.lower() == "baai/bge-large-en-v1.5" else model
        self.base_url = "https://router.huggingface.co"

    def embed_query(self, text: str) -> np.ndarray:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"inputs": text, "options": {"wait_for_model": True}}

        last_error = None
        for template in self.ENDPOINTS:
            url = template.format(model=self.model)
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=60)
            except Exception as e:  # noqa: BLE001 - try the next endpoint
                last_error = str(e)
                continue

            if response.status_code == 200:
                return _l2_normalize(self._to_vector(response.json()))

            last_error = f"{response.status_code}: {response.text[:200]}"
            # 404 means this endpoint shape is wrong; anything else is a real error.
            if response.status_code != 404:
                break

        raise RuntimeError(f"Hugging Face embeddings request failed ({last_error})")

    @staticmethod
    def _to_vector(data) -> np.ndarray:
        """
        HF returns either a flat vector, or token-level vectors that need mean
        pooling, depending on the endpoint. Normalise both shapes to 1024 floats.
        """
        arr = np.array(data, dtype=np.float32)
        while arr.ndim > 1:
            arr = arr.mean(axis=0) if arr.shape[-1] == VECTOR_SIZE else arr[0]
        if arr.shape[-1] != VECTOR_SIZE:
            raise RuntimeError(f"Expected {VECTOR_SIZE}-dim embedding, got shape {arr.shape}")
        return arr


class LocalEmbedder:
    """
    Run BAAI/bge-large-en-v1.5 on this machine via fastembed (ONNX, no PyTorch).

    Free and fast (~42 ms/query) with no API key, and verified to reproduce the
    stored Qdrant vectors exactly (cosine 1.0000). Needs ~2.4 GB RAM, so it is
    for local development, not a 512 MB deployment host.
    """

    def __init__(self, model: str = "BAAI/bge-large-en-v1.5"):
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise ImportError(
                "EMBEDDING_PROVIDER=local needs fastembed. "
                "Install it with: pip install fastembed"
            ) from e

        self.model_name = "BAAI/bge-large-en-v1.5" if model.lower() == "baai/bge-large-en-v1.5" else model
        print(f"Loading local embedding model {self.model_name} (first run downloads ~1.2 GB)...")
        self._model = TextEmbedding(self.model_name)
        print("Local embedding model ready")

    def embed_query(self, text: str) -> np.ndarray:
        vec = next(iter(self._model.query_embed(text)))
        return _l2_normalize(np.array(vec, dtype=np.float32))


def create_embedder(config: RetrievalPipelineConfig):
    """Build the embedder named by config.embedding_provider."""
    provider = (config.embedding_provider or "local").strip().lower()

    if provider == "openrouter":
        if not config.openrouter_api_key:
            raise ValueError("EMBEDDING_PROVIDER=openrouter requires OPENROUTER_API_KEY")
        return OpenRouterEmbedder(api_key=config.openrouter_api_key, model=config.embedding_model)

    if provider in ("huggingface", "hf"):
        if not config.huggingface_api_key:
            raise ValueError(
                "EMBEDDING_PROVIDER=huggingface requires HUGGINGFACE_API_KEY. "
                "Get a free token at https://huggingface.co/settings/tokens"
            )
        return HuggingFaceEmbedder(api_key=config.huggingface_api_key, model=config.embedding_model)

    if provider == "local":
        return LocalEmbedder(model=config.embedding_model)

    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER '{provider}'. Use local, huggingface, or openrouter."
    )


# =============================================================================
# BM25 KEYWORD INDEX
# =============================================================================
class BM25Index:
    """
    BM25 index for keyword search.
    This runs entirely locally - no API calls needed!
    BM25 is good at finding exact keyword matches that semantic search might miss.
    """

    _TOKEN_RE = re.compile(r"[a-z0-9]+")

    def __init__(self, chunks: list[dict], keep_corpus: bool = True):
        """
        Build BM25 index from chunks.

        Args:
            chunks: List of chunk dictionaries from chunks.json
            keep_corpus: When False, the tokenized corpus is released after the
                index is built. The index still works; this just saves ~156 MB.
        """
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.chunk_id_to_idx = {cid: i for i, cid in enumerate(self.chunk_ids)}

        tokenized_docs = [self._tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_docs)

        if keep_corpus:
            self.chunks = chunks
            self.tokenized_docs = tokenized_docs
        else:
            self.chunks = None
            self.tokenized_docs = None
            del tokenized_docs

    def _tokenize(self, text: str) -> list[str]:
        """
        Simple tokenization: lowercase and extract alphanumeric words.

        Args:
            text: Text to tokenize

        Returns:
            List of lowercase word tokens
        """
        return self._TOKEN_RE.findall(text.lower())

    def search(self, query: str, top_k: int = 50) -> list[tuple[int, float]]:
        """
        Search for query and return top-k results.

        Args:
            query: Search query string
            top_k: Maximum number of results to return

        Returns:
            List of (chunk_index, score) tuples, sorted by score descending
        """
        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            (int(idx), float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0
        ]


class CompactBM25Index:
    """
    A memory-lean BM25 index that scores identically to BM25Okapi.

    BM25Okapi keeps one Python dict of term counts per document. Across 11,008
    chunks that costs hundreds of megabytes, and the interpreter does not hand
    the memory back to the OS once it is freed, so a 512 MB host runs out during
    startup. This stores the same postings in flat numpy arrays (about 12 MB)
    and can be built once offline, so the server never parses chunks.json.

    Verified to produce the same top-k ordering as BM25Okapi.
    Build the artifact with:  python build_bm25_index.py
    """

    _TOKEN_RE = re.compile(r"[a-z0-9]+")
    K1 = 1.5
    B = 0.75
    EPSILON = 0.25  # BM25Okapi's floor for negative idf values

    def __init__(self, terms, chunk_ids, offsets, p_docs, p_tfs, idf, doc_len, avgdl):
        self.vocab = {t: i for i, t in enumerate(terms)}
        self.chunk_ids = list(chunk_ids)
        self.chunk_id_to_idx = {cid: i for i, cid in enumerate(self.chunk_ids)}
        self.offsets = offsets
        self.p_docs = p_docs
        self.p_tfs = p_tfs
        self.idf = idf
        self.doc_len = doc_len
        self.avgdl = float(avgdl)
        # Precompute the length-normalisation denominator once.
        self._denom_len = self.K1 * (1 - self.B + self.B * self.doc_len / self.avgdl)
        # Attributes the plain BM25Index also exposes.
        self.chunks = None
        self.tokenized_docs = None

    # -- construction ---------------------------------------------------------
    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        return cls._TOKEN_RE.findall(text.lower())

    @classmethod
    def build(cls, chunks: list[dict]) -> "CompactBM25Index":
        chunk_ids = [c["chunk_id"] for c in chunks]
        n_docs = len(chunks)

        vocab: dict[str, int] = {}
        postings: dict[int, list] = {}
        doc_len = np.zeros(n_docs, dtype=np.int32)

        for did, chunk in enumerate(chunks):
            tokens = cls._tokenize(chunk["text"])
            doc_len[did] = len(tokens)
            tf: dict[int, int] = {}
            for tok in tokens:
                tid = vocab.get(tok)
                if tid is None:
                    tid = vocab[tok] = len(vocab)
                tf[tid] = tf.get(tid, 0) + 1
            for tid, freq in tf.items():
                postings.setdefault(tid, []).append((did, freq))

        n_terms = len(vocab)
        nnz = sum(len(v) for v in postings.values())
        offsets = np.zeros(n_terms + 1, dtype=np.int64)
        p_docs = np.empty(nnz, dtype=np.int32)
        p_tfs = np.empty(nnz, dtype=np.float32)
        idf = np.zeros(n_terms, dtype=np.float32)

        pos = 0
        for tid in range(n_terms):
            plist = postings.get(tid, ())
            offsets[tid] = pos
            for did, freq in plist:
                p_docs[pos] = did
                p_tfs[pos] = freq
                pos += 1
            df = len(plist)
            # Identical to BM25Okapi's raw idf; negatives are floored below.
            idf[tid] = np.log(n_docs - df + 0.5) - np.log(df + 0.5)
        offsets[n_terms] = pos

        idf[idf < 0] = cls.EPSILON * float(idf.mean())

        terms = sorted(vocab, key=vocab.get)
        return cls(terms, chunk_ids, offsets, p_docs, p_tfs, idf, doc_len, float(doc_len.mean()))

    # -- persistence ----------------------------------------------------------
    @staticmethod
    def _pack(strings: list[str]) -> np.ndarray:
        return np.frombuffer("\n".join(strings).encode("utf-8"), dtype=np.uint8)

    @staticmethod
    def _unpack(arr: np.ndarray) -> list[str]:
        return arr.tobytes().decode("utf-8").split("\n")

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            terms=self._pack(sorted(self.vocab, key=self.vocab.get)),
            chunk_ids=self._pack(self.chunk_ids),
            offsets=self.offsets,
            p_docs=self.p_docs,
            p_tfs=self.p_tfs,
            idf=self.idf,
            doc_len=self.doc_len,
            avgdl=np.array([self.avgdl], dtype=np.float64),
        )

    @classmethod
    def load(cls, path: str) -> "CompactBM25Index":
        with np.load(path) as d:
            return cls(
                terms=cls._unpack(d["terms"]),
                chunk_ids=cls._unpack(d["chunk_ids"]),
                offsets=d["offsets"],
                p_docs=d["p_docs"],
                p_tfs=d["p_tfs"],
                idf=d["idf"],
                doc_len=d["doc_len"],
                avgdl=float(d["avgdl"][0]),
            )

    # -- search ---------------------------------------------------------------
    def search(self, query: str, top_k: int = 50) -> list[tuple[int, float]]:
        """Same contract as BM25Index.search: (chunk_index, score), best first."""
        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores = np.zeros(len(self.doc_len), dtype=np.float32)
        for tok in tokens:
            tid = self.vocab.get(tok)
            if tid is None:
                continue
            start, end = self.offsets[tid], self.offsets[tid + 1]
            docs = self.p_docs[start:end]
            freqs = self.p_tfs[start:end]
            scores[docs] += self.idf[tid] * (freqs * (self.K1 + 1)) / (freqs + self._denom_len[docs])

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]


# =============================================================================
# MAIN PIPELINE
# =============================================================================
class RetrievalPipeline:
    """
    Main retrieval pipeline combining semantic search + BM25 + reranking.
    This is what api_server.py uses to find relevant chunks.
    """

    def __init__(self, config: Optional[RetrievalPipelineConfig] = None):
        """
        Initialize all components of the retrieval pipeline.

        Args:
            config: Optional configuration. If None, loads from environment variables.
        """
        if config is None:
            config = RetrievalPipelineConfig(
                qdrant_url=os.getenv("QDRANT_URL"),
                qdrant_api_key=os.getenv("QDRANT_API_KEY"),
                openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
                cohere_api_key=os.getenv("COHERE_API_KEY") or None,
                huggingface_api_key=os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN") or None,
                chunks_path=os.getenv("CHUNKS_PATH") or "./chunks.json",
                bm25_index_path=os.getenv("BM25_INDEX_PATH") or "./bm25_index.npz",
                embedding_model=(os.getenv("EMBEDDING_MODEL") or "baai/bge-large-en-v1.5"),
                embedding_provider=os.getenv("EMBEDDING_PROVIDER") or "local",
                low_memory=_env_bool("LOW_MEMORY", False),
                use_reranker=_env_bool("USE_RERANKER", True)
                and (os.getenv("RERANKER_TYPE") or "cohere").lower() != "none",
            )

        missing = [
            name for name, value in (
                ("QDRANT_URL", config.qdrant_url),
                ("QDRANT_API_KEY", config.qdrant_api_key),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"Missing required configuration: {', '.join(missing)}")

        self.config = config
        self.low_memory = config.low_memory

        # Qdrant Cloud client for semantic search
        self.qdrant = QdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
            timeout=60,
        )

        # Query embedder (must serve the same model used to build the collection)
        self.embedder = create_embedder(config)
        print(f"Embedding provider: {config.embedding_provider}")

        index_path = _resolve_path(config.bm25_index_path)

        if self.low_memory and os.path.exists(index_path):
            # Deployment fast path: load the prebuilt index and never open
            # chunks.json at all. Keeps startup RAM tiny.
            self.bm25_index = CompactBM25Index.load(index_path)
            self.chunks = [{"chunk_id": cid} for cid in self.bm25_index.chunk_ids]
            print(f"Loaded prebuilt BM25 index from {index_path} "
                  f"({len(self.chunks)} chunks, corpus not loaded)")
            return

        # Load chunks (resolve relative to this file if not found from cwd)
        chunks_path = _resolve_path(config.chunks_path)
        print(f"Loading chunks from {chunks_path}...")
        with open(chunks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        chunks: list[dict] = data["chunks"] if isinstance(data, dict) else data
        print(f"Loaded {len(chunks)} chunks")

        if self.low_memory:
            # Same scores as BM25Okapi, a fraction of the memory.
            self.bm25_index = CompactBM25Index.build(chunks)
            self.chunks = [{"chunk_id": cid} for cid in self.bm25_index.chunk_ids]
            del chunks, data
            import gc
            gc.collect()
            print("BM25 index built (compact)")
            print(f"Low-memory mode: corpus released, payloads served from Qdrant. "
                  f"Run 'python build_bm25_index.py' to skip this step at startup.")
        else:
            self.bm25_index = BM25Index(chunks)
            self.chunks = chunks
            print("BM25 index built")

    # -- payload access -------------------------------------------------------
    def _payloads_by_index(self, indices: list[int]) -> dict[int, dict]:
        """
        Get chunk payloads for BM25 hits.

        In normal mode these come from the in-memory corpus. In low-memory mode
        they are fetched from Qdrant, which is safe because upload_from_npz.py
        used the chunk's position in chunks.json as its Qdrant point id, so the
        BM25 index position and the Qdrant point id are the same number.
        """
        if not self.low_memory:
            return {i: self.chunks[i] for i in indices}

        if not indices:
            return {}
        records = self.qdrant.retrieve(
            collection_name=COLLECTION_NAME,
            ids=list(indices),
            with_payload=True,
        )
        return {int(r.id): r.payload for r in records}

    # -- search stages --------------------------------------------------------
    def semantic_search(self, query: str, top_k: int = 30) -> list[dict]:
        """
        Perform semantic search using Qdrant.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of result dicts with chunk_id, score, and payload
        """
        query_embedding = self.embedder.embed_query(query)

        results = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding.tolist(),
            limit=top_k,
            with_payload=True,
        ).points

        return [
            {
                "chunk_id": r.payload["chunk_id"],
                "score": float(r.score),
                "payload": r.payload,
            }
            for r in results
        ]

    def bm25_search(self, query: str, top_k: int = 30) -> list[dict]:
        """
        Perform BM25 keyword search.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of result dicts with chunk_id, score, and payload
        """
        hits = self.bm25_index.search(query, top_k)
        payloads = self._payloads_by_index([idx for idx, _ in hits])

        results = []
        for idx, score in hits:
            payload = payloads.get(idx)
            if payload is None:
                continue
            results.append({
                "chunk_id": payload["chunk_id"],
                "score": score,
                "payload": payload,
            })
        return results

    @staticmethod
    def _normalize(results: list[dict]) -> None:
        """Add a 'normalized_score' (score / max score) to each result in place."""
        if not results:
            return
        max_score = max(r["score"] for r in results)
        for r in results:
            r["normalized_score"] = r["score"] / max_score if max_score > 0 else 0.0

    def hybrid_search(self, query: str, semantic_top_k: int = 30, bm25_top_k: int = 30) -> list[dict]:
        """
        Combine semantic and BM25 results using weighted scoring.

        Args:
            query: Search query
            semantic_top_k: Max results from semantic search
            bm25_top_k: Max results from BM25 search

        Returns:
            Combined and sorted list of results
        """
        try:
            semantic_results = self.semantic_search(query, semantic_top_k)
        except Exception as e:  # noqa: BLE001 - degrade to keyword-only search
            print(f"⚠️  Semantic search unavailable, falling back to BM25 only: {e}")
            semantic_results = []

        bm25_results = self.bm25_search(query, bm25_top_k)

        self._normalize(semantic_results)
        self._normalize(bm25_results)

        sw = self.config.semantic_weight
        bw = self.config.bm25_weight

        combined: dict[str, dict] = {}

        for r in semantic_results:
            combined[r["chunk_id"]] = {
                "chunk_id": r["chunk_id"],
                "payload": r["payload"],
                "semantic_score": r["normalized_score"],
                "bm25_score": 0.0,
                "combined_score": sw * r["normalized_score"],
            }

        for r in bm25_results:
            entry = combined.get(r["chunk_id"])
            if entry is not None:
                entry["bm25_score"] = r["normalized_score"]
                entry["combined_score"] = sw * entry["semantic_score"] + bw * entry["bm25_score"]
            else:
                combined[r["chunk_id"]] = {
                    "chunk_id": r["chunk_id"],
                    "payload": r["payload"],
                    "semantic_score": 0.0,
                    "bm25_score": r["normalized_score"],
                    "combined_score": bw * r["normalized_score"],
                }

        return sorted(combined.values(), key=lambda x: x["combined_score"], reverse=True)

    def rerank(self, query: str, results: list[dict], top_k: int = 10) -> list[dict]:
        """
        Rerank results using Cohere API (optional but improves quality).

        Args:
            query: Original query
            results: Results from hybrid_search
            top_k: Number of results to return after reranking

        Returns:
            Reranked list of results
        """
        if not self.config.cohere_api_key or not results or top_k <= 0:
            return results[:top_k]

        texts = [r["payload"]["text"] for r in results]

        try:
            response = requests.post(
                "https://api.cohere.ai/v2/rerank",
                headers={
                    "Authorization": f"Bearer {self.config.cohere_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.config.rerank_model,
                    "query": query,
                    "documents": texts,
                    "top_n": min(top_k, len(texts)),
                },
                timeout=30,
            )
            if response.status_code != 200:
                raise RuntimeError(f"Cohere rerank failed ({response.status_code}): {response.text[:200]}")

            reranked = []
            for item in response.json()["results"]:
                r = dict(results[item["index"]])
                r["rerank_score"] = float(item["relevance_score"])
                r["score"] = r["rerank_score"]
                reranked.append(r)
            return reranked

        except Exception as e:  # noqa: BLE001 - reranking is optional
            print(f"⚠️  Reranking unavailable, using hybrid order: {e}")
            return results[:top_k]

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        use_hybrid: bool = True,
        use_reranker: Optional[bool] = None,
    ) -> list[RetrievalResult]:
        """
        Full retrieval pipeline - THIS IS WHAT api_server.py CALLS!

        Args:
            query: User's search query
            top_k: Number of results to return
            use_hybrid: If False, use semantic search only (no BM25 fusion)
            use_reranker: Override the configured reranker setting for this call

        Returns:
            List of RetrievalResult objects ready for RAG generation
        """
        if use_hybrid:
            candidates = self.hybrid_search(query)
        else:
            candidates = self.semantic_search(query, top_k=max(top_k * 4, 30))
            for c in candidates:
                c["combined_score"] = c["score"]

        if not candidates:
            return []

        rerank_enabled = self.config.use_reranker if use_reranker is None else use_reranker
        if rerank_enabled:
            reranked = self.rerank(query, candidates, top_k=min(top_k * 2, len(candidates)))
        else:
            reranked = candidates

        final = reranked[:top_k]

        return [
            RetrievalResult(
                chunk_id=r["payload"]["chunk_id"],
                paper_id=r["payload"]["paper_id"],
                title=r["payload"]["title"],
                authors=r["payload"]["authors"],
                text=r["payload"]["text"],
                score=float(r.get("rerank_score", r.get("combined_score", r.get("score", 0)))),
                chunk_type=r["payload"].get("chunk_type", ""),
                chunk_section=r["payload"].get("chunk_section", ""),
                pdf_url=r["payload"].get("pdf_url"),
                github_link=r["payload"].get("github_link"),
                video_link=r["payload"].get("video_link"),
                acm_url=r["payload"].get("acm_url"),
                abstract_url=r["payload"].get("abstract_url"),
            )
            for r in final
        ]


# For testing this file directly
if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "3D Gaussian Splatting"

    print(f"Testing retrieval pipeline with query: '{query}'")
    print("=" * 60)

    pipeline = RetrievalPipeline()
    results = pipeline.retrieve(query, top_k=5)

    print(f"\nFound {len(results)} results:\n")

    for i, r in enumerate(results, 1):
        print(f"{i}. [{r.score:.4f}] {r.title[:60]}...")
        print(f"   Paper ID: {r.paper_id}")
        print(f"   Text preview: {r.text[:100]}...")
        print()
