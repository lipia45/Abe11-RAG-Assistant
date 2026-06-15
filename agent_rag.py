"""
Phase D - Multi-agent RAG with tool/function calling + query routing.

Two-agent workflow (Claude Agent SDK / Messages API + tool use):
  1. ROUTER agent  - classifies whether a query needs the knowledge base.
  2. ANSWER agent  - if so, runs a tool-use loop: calls `search_knowledge_base`,
     reads the returned passages, and synthesizes a CITED answer. If not, it
     answers directly / politely declines out-of-scope chit-chat.

Reuses the Phase A winning retrieval (chunk900_150, cached embeddings).
Global cost ledger enforces the $15 cap across all phases.

Run: `py agent_rag.py`  (runs a short demo on mixed queries).
Importable: `RagAgent().answer(question)`.
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json, time, re
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "eval" / "results"; PROC = ROOT / "data" / "processed"; CACHE = ROOT / "data" / "cache"
PROGRESS = RESULTS / "progress.log"; LEDGER = CACHE / "cost_ledger.json"
load_dotenv(ROOT / ".env")

MODEL = "claude-haiku-4-5"; PRICE_IN, PRICE_OUT = 1.0, 5.0
COST_CAP_USD = 15.0
CHUNK_NAME = "chunk900_150"; CHUNK_SIZE, CHUNK_OVERLAP = 900, 150
EMB_MODEL = "BAAI/bge-small-en-v1.5"; TOPK = 5

t0 = time.time()
_ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {"total_usd": 0.0, "phases": {}}
_cost = {"in": 0, "out": 0}


def _log(m):
    line = f"[{time.time()-t0:7.1f}s] {m}"
    print(line, flush=True)
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _spent():
    return _ledger["total_usd"] + _cost["in"] / 1e6 * PRICE_IN + _cost["out"] / 1e6 * PRICE_OUT


def _track(usage):
    _cost["in"] += usage.input_tokens; _cost["out"] += usage.output_tokens
    if _spent() >= COST_CAP_USD:
        raise SystemExit(f"GLOBAL COST CAP ${COST_CAP_USD} reached - aborting.")


def _chunk(text, size, overlap):
    if len(text) <= size:
        return [text]
    step = size - overlap
    return [c for c in (text[i:i + size] for i in range(0, len(text), step)) if c.strip()]


class Retriever:
    """Loads the Phase A winning index from cached embeddings (no re-embed)."""
    def __init__(self):
        corpus = pd.read_parquet(PROC / "corpus.parquet")
        self.chunk_txt, self.parent = [], []
        for cid, txt in zip(corpus["context_id"], corpus["context"]):
            for ch in _chunk(txt, CHUNK_SIZE, CHUNK_OVERLAP):
                self.chunk_txt.append(ch); self.parent.append(cid)
        from sentence_transformers import SentenceTransformer
        import torch
        self.embedder = SentenceTransformer(EMB_MODEL,
                                            device="cuda" if torch.cuda.is_available() else "cpu")
        cache = CACHE / f"{CHUNK_NAME}_emb.npy"
        if cache.exists():
            vecs = np.load(cache)
        else:  # first run: embed the corpus once and cache it (runnable from scratch)
            CACHE.mkdir(parents=True, exist_ok=True)
            _log(f"[retriever] no cache - embedding {len(self.chunk_txt)} chunks (one-time) ...")
            vecs = self.embedder.encode(self.chunk_txt, normalize_embeddings=True,
                                        batch_size=128, show_progress_bar=True).astype("float32")
            np.save(cache, vecs)
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        self.qc = QdrantClient(":memory:")
        self.qc.create_collection("c", vectors_config=VectorParams(size=vecs.shape[1],
                                                                   distance=Distance.COSINE))
        for b in range(0, len(self.chunk_txt), 2000):
            e = min(b + 2000, len(self.chunk_txt))
            self.qc.upsert("c", points=[PointStruct(id=i, vector=vecs[i].tolist(),
                                                    payload={"cid": self.parent[i], "text": self.chunk_txt[i]})
                                        for i in range(b, e)])
        _log(f"[retriever] index ready ({len(self.chunk_txt)} chunks)")

    def search(self, query, k=TOPK):
        qv = self.embedder.encode([query], normalize_embeddings=True)[0]
        res = self.qc.query_points("c", query=qv.tolist(), limit=k * 3).points
        out, seen = [], set()
        for p in res:
            if p.payload["cid"] not in seen:
                seen.add(p.payload["cid"]); out.append({"id": p.payload["cid"], "text": p.payload["text"]})
            if len(out) >= k:
                break
        return out


SEARCH_TOOL = {
    "name": "search_knowledge_base",
    "description": "Search the private document collection for passages relevant to a query. "
                   "Returns up to k passages, each with an ID like ctx_000123 and its text.",
    "input_schema": {"type": "object", "properties": {
        "query": {"type": "string", "description": "the search query"},
        "k": {"type": "integer", "description": "number of passages (default 5)"}},
        "required": ["query"]},
}
ANSWER_SYS = (
    "You answer questions over a private knowledge base. Use the search_knowledge_base tool to "
    "find relevant passages, then answer using ONLY those passages, citing the passage IDs inline "
    "in square brackets like [ctx_000123]. If the passages do not contain the answer, say you "
    "don't know based on the knowledge base.")
ROUTER_SCHEMA = {"type": "object", "properties": {
    "needs_retrieval": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["needs_retrieval", "reason"], "additionalProperties": False}


class RagAgent:
    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic()
        self.retriever = Retriever()

    def _route(self, question):
        m = self.client.messages.create(
            model=MODEL, max_tokens=120,
            system="Decide if answering this user message requires searching a private document "
                   "knowledge base (factual/informational questions) or not (greetings, chit-chat, "
                   "meta questions about you).",
            messages=[{"role": "user", "content": question}],
            output_config={"format": {"type": "json_schema", "schema": ROUTER_SCHEMA}})
        _track(m.usage)
        return json.loads(next(b.text for b in m.content if b.type == "text"))

    def _answer_with_tools(self, question):
        messages = [{"role": "user", "content": question}]
        tool_calls = 0
        while True:
            resp = self.client.messages.create(model=MODEL, max_tokens=600, system=ANSWER_SYS,
                                               tools=[SEARCH_TOOL], messages=messages)
            _track(resp.usage)
            if resp.stop_reason != "tool_use":
                return next((b.text for b in resp.content if b.type == "text"), ""), tool_calls
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for b in resp.content:
                if b.type == "tool_use":
                    tool_calls += 1
                    hits = self.retriever.search(b.input["query"], b.input.get("k", TOPK))
                    payload = "\n\n".join(f"[{h['id']}] {h['text']}" for h in hits)
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": payload})
            messages.append({"role": "user", "content": results})

    def answer(self, question):
        route = self._route(question)
        if not route["needs_retrieval"]:
            m = self.client.messages.create(model=MODEL, max_tokens=200,
                                            system="You are a helpful assistant for a RAG knowledge "
                                                   "base. For chit-chat, reply briefly and note you "
                                                   "answer questions about the document collection.",
                                            messages=[{"role": "user", "content": question}])
            _track(m.usage)
            return {"route": "direct", "answer": next((b.text for b in m.content if b.type == "text"), ""),
                    "tool_calls": 0}
        ans, tc = self._answer_with_tools(question)
        return {"route": "retrieval", "answer": ans, "tool_calls": tc}


def _save_ledger():
    _ledger["phases"]["D_agent"] = round(_ledger["phases"].get("D_agent", 0)
                                         + _cost["in"] / 1e6 * PRICE_IN + _cost["out"] / 1e6 * PRICE_OUT, 4)
    _ledger["total_usd"] = round(_spent(), 4)
    LEDGER.write_text(json.dumps(_ledger))


if __name__ == "__main__":
    PROGRESS.write_text("", encoding="utf-8")
    _log(f"START Phase D - multi-agent RAG demo. prior spend ${_ledger['total_usd']:.4f} cap ${COST_CAP_USD}")
    agent = RagAgent()
    # mixed demo: 5 knowledge questions from the test set + 1 chit-chat (routing)
    test = pd.read_parquet(PROC / "eval.parquet")
    test = test[(test["split"] == "test") & test["question"].notna()].sample(n=5, random_state=7)
    demo_qs = test["question"].tolist() + ["Hi! Who are you and what can you do?"]
    transcript = []
    for q in demo_qs:
        r = agent.answer(q)
        _log(f"--- Q: {q[:90]}")
        _log(f"    route={r['route']} tool_calls={r['tool_calls']}")
        _log(f"    A: {r['answer'][:260]}")
        transcript.append({"question": q, **r})
    _save_ledger()
    (RESULTS / "agent_demo.json").write_text(json.dumps(transcript, indent=2, ensure_ascii=False))
    _log(f"PHASE D DONE | demo saved -> agent_demo.json | GLOBAL TOTAL ${_ledger['total_usd']:.4f} / ${COST_CAP_USD}")
