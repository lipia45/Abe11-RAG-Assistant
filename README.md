# Multi-Agent RAG Knowledge Assistant

A retrieval-augmented generation (RAG) system that answers questions over a private
document collection with **cited answers**, built as a **multi-agent workflow** with
tool/function calling and an MCP server. Every component is measured: retrieval and
answer quality are evaluated on a held-out test split, and every Claude API call is
cost-tracked against a hard budget cap.

![Architecture](docs/architecture.png)

## What it does

- **Retrieval** — chunk documents, embed with `bge-small` (GPU), index in **Qdrant**,
  retrieve the top-k relevant passages for a query.
- **Multi-agent answering** — a **router agent** decides whether a query needs the
  knowledge base; an **answer agent** uses a `search_knowledge_base` **tool** (function
  calling) to fetch passages and synthesizes an answer that **cites passage IDs**.
- **MCP server** — exposes the retriever as a Model Context Protocol tool so any MCP
  client (Claude Desktop, Agent SDK) can use it.
- **Evaluation** — retrieval metrics (Recall@k, Precision@k, nDCG@k, MRR) and answer
  quality (RAG Triad via LLM-judge + answer correctness vs gold), tuned on train,
  reported on test.

## Tech stack

Python · **Qdrant** (vector DB, cosine) · `sentence-transformers` **bge-small-en-v1.5**
(GPU) · **Claude** (Haiku 4.5) via the Anthropic SDK with **tool use** · **MCP**
(FastMCP) · pandas/NumPy · matplotlib + D2 (figures).

## Dataset

[`neural-bridge/rag-dataset-12000`](https://huggingface.co/datasets/neural-bridge/rag-dataset-12000)
— 12,000 (context, question, answer) triples, 9,600 train / 2,400 test, Apache 2.0.
Contexts form the corpus (haystack); the question/gold-answer pairs form the eval set.

## Results (measured, held-out test)

**Retrieval** — tuned on train, reported on test. A chunking sweep selected `chunk 900/150`:

| Config | Recall@1 | Recall@5 | Recall@10 | nDCG@10 | MRR | operating k |
|---|---|---|---|---|---|---|
| Whole context (baseline) | 0.815 | 0.896 | 0.913 | 0.864 | 0.851 | 10 |
| **Chunk 900/150 (chosen)** | **0.841** | **0.907** | **0.931** | **0.886** | **0.873** | **5** |

Chunking lifted every metric **and** halved the operating cutoff (k 10 -> 5), so the
generator receives half the context at higher recall and ~2x precision.

The winning config was re-evaluated on the **full 2,400-question test set** (Recall@1
0.845, Recall@5 0.909, Recall@10 0.928, nDCG@10 0.886, MRR 0.874) — within 0.3% of the
sample, and train (9,600) ≈ test (2,400) confirms no overfit.

**Generation (300 test questions, cited answers):** 92.3% of answers cite a retrieved
passage; 90.0% have the gold context in the top-5.

**Answer quality:** answer correctness vs gold (embedding cosine) = **0.857**;
RAG Triad (LLM-judge) faithfulness **0.797**, answer relevance **0.706**, context
relevance **0.657**.

![Answer quality](docs/fig-answer-quality.png)

## Benchmark validation against BEIR

To validate that the retrieval pipeline is correctly implemented against the field
standard, the same `bge-small-en-v1.5` embedder was benchmarked on
[BEIR](https://github.com/beir-cellar/beir) (Thakur et al., NeurIPS 2021) via the `mteb`
library — reproducing the published MTEB protocol.

| BEIR task | Ours (nDCG@10) | Published bge-small | |
|---|---|---|---|
| SciFact | **0.7127** | 0.713 | exact match |
| NFCorpus | **0.3430** | ~0.343 | match |
| ArguAna | **0.6035** | ~0.593 | match |

![BEIR: ours vs published](docs/fig-beir-validation.png)

Reproducing the published scores (SciFact essentially exact) confirms the retriever is
configured to the standard protocol — which in turn validates the metric numbers reported
above on the project corpus.

## End-to-end robustness on RGB (standard benchmark)

Beyond retrieval, the full answer pipeline was evaluated on the **RGB benchmark**
(Chen et al., AAAI 2024) — a standard RAG benchmark that mixes relevant and noise
documents and scores whether the generated answer is correct. Run on the standard
English set (300 questions); generator = Claude Haiku 4.5.

| Noise ratio | Answer accuracy (ours) |
|---|---|
| 0.0 | 0.983 |
| 0.4 | 0.973 |
| 0.8 | **0.937** |

RGB is designed to break down at high noise: the paper's ChatGPT baseline falls to
**0.760** at 0.8 noise, while this pipeline holds **0.937** — substantially stronger
noise robustness.

![RGB noise robustness](docs/fig-rgb-noise-robustness.png)

## Pipeline / how to run

```bash
pip install -r requirements.txt          # or: py -m pip install -r requirements.txt
cp .env.example .env                      # then put your real ANTHROPIC_API_KEY in .env
```

| Step | Command | Cost |
|---|---|---|
| 1. Prepare the corpus | `py scripts/01_prepare_data.py` | free |
| 2. Query — retrieval only | `py scripts/03_query.py --retrieve-only "your question"` | free |
| 3. Query — full cited answer | `py scripts/03_query.py "your question"` | ~$0.002 |
| 4. Multi-agent demo (routing + tool use) | `py agent_rag.py` | ~$0.03 |
| MCP server | `py mcp_server.py` | free |

On the first query the index is built from the corpus and cached automatically
(`bge-small` uses the GPU when a CUDA build of PyTorch is installed, else CPU). The
evaluation/experiment scripts that produced the numbers above (chunking sweep, generation
eval, RAGAS, BEIR) are kept outside this repo to keep it focused on the reusable system.

## Configuration / secrets

Secrets are read from a local `.env` (git-ignored), never hardcoded:

```
ANTHROPIC_API_KEY=your_key_here
```

`.env.example` is the committed template with placeholders only. The code reads
`os.environ["ANTHROPIC_API_KEY"]` via `python-dotenv` — the key value never appears in
any committed file.

## Logging, observability & cost control

- Every long run streams progress to `eval/results/progress.log` (tail it live).
- Every Claude call is token- and cost-tracked; a global ledger
  (`data/cache/cost_ledger.json`) enforces a **hard $15 cap across all phases** — runs
  abort before exceeding it.
- All LLM outputs are logged (`data/processed/generations.parquet`,
  `eval/results/phase_c_scores.parquet`, `agent_demo.json`) for inspection.
- **Total spend to build the full evaluated system: ~$0.82** (Claude Haiku 4.5).

## Limitations (honest)

- The dataset's questions are derived from their contexts, so retrieval is relatively
  easy (Recall@1 ~0.84); a harder, multi-relevant corpus would stress retrieval more.
- qrels are single-relevant (one gold context per question), so Recall@k is a lower
  bound and equals Hit@k.
- Context relevance (0.657) is the weakest dimension — feeding 5 chunks per query adds
  noise; a cross-encoder reranker is the clear next improvement.
- Metrics are computed on samples (800 retrieval queries; 300 generations; 150 judged)
  with `bge-small`; larger samples / a stronger embedder would tighten the estimates.
- A hallucination-leaderboard comparison (Vectara HHEM) was attempted but blocked by a
  model/transformers incompatibility; faithfulness is instead reported via the RAG Triad.
