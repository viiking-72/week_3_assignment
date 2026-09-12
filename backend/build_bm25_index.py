#!/usr/bin/env python3
"""
Build the compact BM25 index used by the deployed backend.

Why this exists
---------------
BM25Okapi keeps one Python dict of term counts per document. Across 11,008
chunks that costs hundreds of megabytes at startup, and CPython does not return
that memory to the OS once it is freed. On a 512 MB host (Render's free tier)
the process is killed before it ever serves a request.

This script does that work once, here on your machine, and writes a small
artifact of flat numpy arrays. The server then loads the artifact and never
opens chunks.json at all. Scoring is unchanged: the compact index reproduces
BM25Okapi's ranking exactly.

Usage:
    python build_bm25_index.py                  # writes ./bm25_index.npz
    python build_bm25_index.py --verify         # also check it matches BM25Okapi
    python build_bm25_index.py --out other.npz
"""

import argparse
import json
import os
import time

from retrieval_pipeline import CompactBM25Index, _resolve_path

DEFAULT_CHUNKS = os.getenv("CHUNKS_PATH") or "./chunks.json"
DEFAULT_OUT = os.getenv("BM25_INDEX_PATH") or "./bm25_index.npz"


def main():
    parser = argparse.ArgumentParser(description="Build the compact BM25 index")
    parser.add_argument("--chunks", default=DEFAULT_CHUNKS, help="Path to chunks.json")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output .npz path")
    parser.add_argument("--verify", action="store_true",
                        help="Compare rankings against rank_bm25's BM25Okapi")
    args = parser.parse_args()

    chunks_path = _resolve_path(args.chunks)
    print("=" * 60)
    print("Building compact BM25 index")
    print("=" * 60)
    print(f"Chunks: {chunks_path}")

    with open(chunks_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = data["chunks"] if isinstance(data, dict) else data
    print(f"Loaded {len(chunks)} chunks")

    t0 = time.time()
    index = CompactBM25Index.build(chunks)
    print(f"Index built in {time.time() - t0:.1f}s "
          f"({len(index.vocab)} terms, {len(index.p_docs)} postings)")

    if args.verify:
        print("\nVerifying against rank_bm25.BM25Okapi...")
        import numpy as np
        from rank_bm25 import BM25Okapi

        tokenized = [CompactBM25Index._tokenize(c["text"]) for c in chunks]
        reference = BM25Okapi(tokenized)

        queries = [
            "3D Gaussian Splatting",
            "cloth simulation speed",
            "neural rendering denoising",
            "differentiable fluid simulation",
        ]
        all_match = True
        for q in queries:
            ref_top = np.argsort(reference.get_scores(CompactBM25Index._tokenize(q)))[::-1][:10]
            new_top = [i for i, _ in index.search(q, top_k=10)]
            match = list(ref_top) == list(new_top)
            all_match &= match
            print(f"  {q!r:36s} identical top-10 order: {match}")
        if not all_match:
            raise SystemExit("✗ Compact index does not match BM25Okapi - not written")
        print("✓ Rankings identical")

    index.save(args.out)
    out_path = args.out if args.out.endswith(".npz") else args.out + ".npz"
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\n✓ Wrote {out_path} ({size_mb:.1f} MB)")
    print("\nThe deployed backend loads this instead of chunks.json.")
    print("Set LOW_MEMORY=true in the deploy environment to use it.")
    print("=" * 60)


if __name__ == "__main__":
    main()
