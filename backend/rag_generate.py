#!/usr/bin/env python3
"""
RAG Generation Pipeline for SIGGRAPH 2025 Papers.

Uses the retrieval pipeline to find relevant chunks,
then generates an answer using an LLM.

The LLM provider is pluggable. Set LLM_PROVIDER in .env:

    cohere      - Cohere's OpenAI-compatible endpoint. The trial key is free,
                  needs no credit card, and is capped at 1,000 calls/month,
                  which bounds spend instead of leaving it open-ended.
    openrouter  - OpenRouter (original assignment path, needs credits).

Both speak the same request and response shape, including streaming, so
api_server.py streams from either without special cases.

Usage:
    from rag_generate import RAGGenerator, GenerationConfig, SYSTEM_PROMPT

    generator = RAGGenerator()
    result = generator.generate("What is 3D Gaussian Splatting?")
    print(result["answer"])
"""

import os
import requests
from typing import Optional
from dataclasses import dataclass, field

from dotenv import load_dotenv
load_dotenv()

from retrieval_pipeline import RetrievalPipeline, RetrievalResult


# =============================================================================
# SYSTEM PROMPT - This tells the LLM how to behave
# =============================================================================
SYSTEM_PROMPT = """You are an expert research assistant specializing in computer graphics, specifically SIGGRAPH 2025 papers.

Your task is to answer questions using ONLY the provided research paper excerpts.

Rules:
1. Cite sources using [Paper Title] format
2. Be comprehensive and technically accurate
3. If the excerpts don't contain the answer, say so
4. Use LaTeX for math: $inline$ or $$block$$
5. Do NOT make up information not in the excerpts
6. Do NOT include a References section at the end
"""


# =============================================================================
# QUERY REFINEMENT PROMPT
# =============================================================================
QUERY_REFINEMENT_PROMPT = """You are an expert at refining search queries for academic paper retrieval.

Given a user's question, rewrite it as a clear, focused search query that will retrieve the most relevant research papers.

Keep it concise (under 20 words). Focus on key technical terms.

User question: {query}

Refined search query:"""


# =============================================================================
# PROVIDER REGISTRY
# =============================================================================
# Each provider exposes an OpenAI-shaped /chat/completions endpoint.
PROVIDERS = {
    "cohere": {
        "base_url": "https://api.cohere.ai/compatibility/v1",
        "api_key_env": "COHERE_API_KEY",
        "default_model": "command-a-03-2025",
        "default_refinement_model": "command-r7b-12-2024",
        "model_prefix": "",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "default_model": "gpt-4o-mini",
        "default_refinement_model": "gpt-4o-mini",
        "model_prefix": "openai/",
    },
}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def _resolve_provider(name: Optional[str]) -> str:
    """Map a configured provider name onto a key in PROVIDERS."""
    value = (name or "").strip().lower()
    # api_server historically passed llm_provider="openai", meaning "the
    # OpenAI-family model served through OpenRouter".
    if value in ("", "openai"):
        value = (os.getenv("LLM_PROVIDER") or "cohere").strip().lower()
    if value in ("", "openai"):
        value = "openrouter"
    if value not in PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{value}'. Use one of: {', '.join(PROVIDERS)}"
        )
    return value


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class GenerationConfig:
    """
    Configuration for the RAG generator.

    Defaults come from the environment (.env) so the same settings drive both
    this module and api_server.py.
    """
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "cohere"))
    llm_model: Optional[str] = field(default_factory=lambda: os.getenv("LLM_MODEL") or None)
    temperature: float = field(default_factory=lambda: _env_float("TEMPERATURE", 0.1))
    max_tokens: int = field(default_factory=lambda: _env_int("MAX_TOKENS", 2000))
    openrouter_api_key: Optional[str] = None  # Loaded from env if not set
    refine_query: bool = field(default_factory=lambda: _env_bool("REFINE_QUERY", True))
    refinement_model: Optional[str] = field(default_factory=lambda: os.getenv("REFINEMENT_MODEL") or None)
    retrieval_top_k: int = field(default_factory=lambda: _env_int("RETRIEVAL_TOP_K", 8))
    use_reranker: bool = field(default_factory=lambda: _env_bool("USE_RERANKER", True))

    def __post_init__(self):
        self.llm_provider = _resolve_provider(self.llm_provider)
        spec = PROVIDERS[self.llm_provider]

        if not self.llm_model:
            self.llm_model = spec["default_model"]
        if not self.refinement_model:
            self.refinement_model = spec["default_refinement_model"]

        # Accept a model written either bare or already prefixed.
        prefix = spec["model_prefix"]
        if prefix and self.llm_model.startswith(prefix):
            self.llm_model = self.llm_model[len(prefix):]

    @property
    def base_url(self) -> str:
        return PROVIDERS[self.llm_provider]["base_url"]

    @property
    def llm_model_id(self) -> str:
        """Fully qualified model id for the active provider."""
        spec = PROVIDERS[self.llm_provider]
        prefix = spec["model_prefix"]
        if not prefix or "/" in self.llm_model:
            return self.llm_model
        return f"{prefix}{self.llm_model}"

    @property
    def refinement_model_id(self) -> str:
        spec = PROVIDERS[self.llm_provider]
        prefix = spec["model_prefix"]
        if not prefix or "/" in self.refinement_model:
            return self.refinement_model
        return f"{prefix}{self.refinement_model}"


# =============================================================================
# RAG GENERATOR CLASS
# =============================================================================
class RAGGenerator:
    """
    Main RAG class - this is what api_server.py uses!

    Flow:
    1. Refine the user's query (optional)
    2. Retrieve relevant chunks using the retrieval pipeline
    3. Format chunks into context
    4. Generate answer using LLM
    5. Return answer with source metadata
    """

    def __init__(self, config: Optional[GenerationConfig] = None, retrieval_pipeline=None):
        """
        Initialize the RAG generator.

        Args:
            config: Optional configuration object
            retrieval_pipeline: Optional pre-initialized retrieval pipeline
        """
        self.config = config or GenerationConfig()

        spec = PROVIDERS[self.config.llm_provider]
        self.llm_base_url = spec["base_url"]
        self.llm_api_key = (
            self.config.openrouter_api_key
            or os.getenv(spec["api_key_env"])
        )
        if not self.llm_api_key:
            raise ValueError(
                f"{spec['api_key_env']} not set (required for LLM_PROVIDER="
                f"{self.config.llm_provider})"
            )

        # Aliases kept so any code written against the original OpenRouter-only
        # attribute names keeps working, whichever provider is active.
        self.openrouter_api_key = self.llm_api_key
        self.openrouter_base_url = self.llm_base_url

        self.retrieval = retrieval_pipeline or RetrievalPipeline()
        # Keep the retrieval pipeline's reranker setting in sync with our config.
        self.retrieval.config.use_reranker = self.config.use_reranker

        print(f"LLM provider: {self.config.llm_provider} ({self.config.llm_model_id})")

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.llm_api_key}",
            "Content-Type": "application/json",
        }

    def refine_query(self, query: str) -> str:
        """
        Use LLM to improve the search query (optional but helps retrieval).

        Args:
            query: Original user query

        Returns:
            Refined query (or original if refinement disabled/fails)
        """
        if not self.config.refine_query:
            return query

        prompt = QUERY_REFINEMENT_PROMPT.format(query=query)
        payload = {
            "model": self.config.refinement_model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 100,
        }

        try:
            response = requests.post(
                f"{self.llm_base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=30,
            )
            if response.status_code != 200:
                print(f"⚠️  Query refinement failed ({response.status_code}); using original query")
                return query

            refined = response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:  # noqa: BLE001 - refinement is best-effort
            print(f"⚠️  Query refinement error; using original query: {e}")
            return query

        # Strip surrounding quotes and any leading label the model may echo back.
        refined = refined.strip().strip('"').strip("'").strip()
        if refined.lower().startswith("refined search query:"):
            refined = refined.split(":", 1)[1].strip().strip('"').strip("'")

        return refined or query

    def _format_context(self, results: list[RetrievalResult]) -> str:
        """
        Format retrieved chunks into a context string for the LLM.

        Args:
            results: List of RetrievalResult objects

        Returns:
            Formatted context string
        """
        formatted_sources = []
        for i, result in enumerate(results, 1):
            formatted_sources.append(
                f"--- Source {i} ---\n"
                f"Title: {result.title}\n"
                f"Authors: {result.authors}\n"
                f"Section: {result.chunk_section or result.chunk_type or 'N/A'}\n"
                f"\n"
                f"Content:\n"
                f"{result.text}\n"
            )
        return "\n".join(formatted_sources)

    def _build_sources_metadata(self, results: list[RetrievalResult]) -> list[dict]:
        """
        Build list of unique source papers for citations.
        The frontend displays these as clickable source links.

        Args:
            results: List of RetrievalResult objects

        Returns:
            List of unique source metadata dicts (deduplicated by title,
            preserving retrieval order)
        """
        seen: dict[str, dict] = {}
        for result in results:
            if result.title not in seen:
                seen[result.title] = {
                    "title": result.title,
                    "authors": result.authors,
                    "pdf_url": result.pdf_url,
                    "github_link": result.github_link,
                    "video_link": result.video_link,
                    "acm_url": result.acm_url,
                    "abstract_url": result.abstract_url,
                }
        return list(seen.values())

    def _build_messages(self, query: str, context: str) -> list[dict]:
        user_message = (
            "Based on the following research paper excerpts, answer this question.\n\n"
            f"Question: {query}\n\n"
            "Research Paper Excerpts:\n"
            f"{context}\n\n"
            "Remember to cite papers using [Paper Title] format."
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

    def _call_llm(self, query: str, context: str) -> str:
        """
        Call the configured provider's chat API to generate an answer.

        Args:
            query: User's question
            context: Formatted context from retrieved chunks

        Returns:
            Generated answer string
        """
        payload = {
            "model": self.config.llm_model_id,
            "messages": self._build_messages(query, context),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        response = requests.post(
            f"{self.llm_base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=120,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"{self.config.llm_provider} chat request failed "
                f"({response.status_code}): {response.text[:300]}"
            )

        return response.json()["choices"][0]["message"]["content"]

    def generate(self, query: str, top_k: Optional[int] = None, return_sources: bool = True) -> dict:
        """
        Full RAG pipeline - retrieve relevant chunks and generate an answer.
        THIS IS THE MAIN METHOD THAT api_server.py CALLS!

        Args:
            query: User's question
            top_k: Number of chunks to retrieve (uses config default if None)
            return_sources: Whether to include source metadata

        Returns:
            Dict with query, refined_query, answer, and sources
        """
        top_k = top_k or self.config.retrieval_top_k

        refined = self.refine_query(query)

        results = self.retrieval.retrieve(refined, top_k=top_k)

        if not results:
            return {
                "query": query,
                "refined_query": refined,
                "answer": "I couldn't find any relevant papers to answer this question.",
                "sources": [],
            }

        context = self._format_context(results)
        answer = self._call_llm(query, context)

        return {
            "query": query,
            "refined_query": refined,
            "answer": answer,
            "sources": self._build_sources_metadata(results) if return_sources else [],
        }


# =============================================================================
# CLI FOR TESTING
# =============================================================================
if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "What is 3D Gaussian Splatting?"

    print("Initializing RAG Generator...")
    generator = RAGGenerator()

    print(f"\nQuery: {query}")
    print("=" * 60)

    result = generator.generate(query)

    print(f"Refined Query: {result.get('refined_query', 'N/A')}")
    print("=" * 60)
    print("\nAnswer:")
    print(result['answer'])
    print("=" * 60)
    print(f"\nSources: {len(result.get('sources', []))} papers")
    for source in result.get('sources', []):
        print(f"  - {source['title']}")
