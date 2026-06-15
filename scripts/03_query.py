"""CLI to query the RAG system.

Default: full multi-agent answer (route -> retrieve -> cited answer) via Claude
(requires ANTHROPIC_API_KEY in .env).
`--retrieve-only`: just print the retrieved passages (free, no API call).

Usage:
    py scripts/03_query.py "your question here"
    py scripts/03_query.py --retrieve-only --top-k 5 "your question here"

Prerequisite: run `py scripts/01_prepare_data.py` first to produce the corpus.
On the first query the index is built and cached automatically.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))  # import the top-level agent module


def main() -> None:
    ap = argparse.ArgumentParser(description="Query the RAG knowledge base.")
    ap.add_argument("question")
    ap.add_argument("--retrieve-only", action="store_true",
                    help="only show retrieved passages (no LLM call, free)")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    if args.retrieve_only:
        from agent_rag import Retriever
        for rank, hit in enumerate(Retriever().search(args.question, args.top_k), 1):
            print(f"[{rank}] {hit['id']}\n    {hit['text'][:300]}\n")
        return

    from agent_rag import RagAgent
    res = RagAgent().answer(args.question)
    print(f"route: {res['route']} | tool_calls: {res['tool_calls']}\n")
    print(res["answer"])


if __name__ == "__main__":
    main()
