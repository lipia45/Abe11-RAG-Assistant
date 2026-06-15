"""
01_prepare_data.py

Self-contained, inspectable data-preparation pipeline for the RAG project.

Source dataset : neural-bridge/rag-dataset-12000 (HuggingFace)
License        : Apache 2.0 (the underlying contexts follow Falcon RefinedWeb /
                 ODC-By terms; see DATA_QUALITY.md).

This script does NOT import any project src/ module on purpose: it must stay
runnable and inspectable on its own. Every stage prints an intermediate
checkpoint so it can be verified and fixed independently.

Stages
  0. Load dataset (download/cache via `datasets`).
  1. Flatten splits into a single pandas DataFrame (+ split column).
  2. Profile: row counts, char-length stats, null/empty, duplicates, uniques.
  3. Build corpus.parquet   : one row per UNIQUE context, stable id "ctx_000001".
  4. Build eval.parquet     : [question, answer, gold_context_id, split].
  5. Write DATA_QUALITY.md  : human-readable report + examples + caveats.
  6. Print stdout summary.

Outputs
  data/processed/corpus.parquet
  data/processed/eval.parquet
  data/DATA_QUALITY.md
"""

from __future__ import annotations

import sys
import io
from pathlib import Path
from datetime import datetime, timezone

# Windows consoles often default to a legacy code page (e.g. cp874) that cannot
# encode characters present in web-scraped text (copyright sign, smart quotes).
# Force UTF-8 on stdout/stderr so previews never crash the pipeline.
for _stream in ("stdout", "stderr"):
    _s = getattr(sys, _stream)
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        setattr(sys, _stream,
                io.TextIOWrapper(_s.buffer, encoding="utf-8", errors="replace"))

import pandas as pd

try:
    from datasets import load_dataset
except ImportError:
    sys.stderr.write(
        "ERROR: the `datasets` package is required. Install with:\n"
        "    py -m pip install datasets pandas pyarrow\n"
    )
    raise

# --------------------------------------------------------------------------
# Paths (resolved relative to this file so the script is location-stable)
# --------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent                      # .../rag_assistant
DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CORPUS_PATH = PROCESSED_DIR / "corpus.parquet"
EVAL_PATH = PROCESSED_DIR / "eval.parquet"
REPORT_PATH = DATA_DIR / "DATA_QUALITY.md"

DATASET_NAME = "neural-bridge/rag-dataset-12000"
FIELDS = ["context", "question", "answer"]

SEP = "-" * 70


def banner(msg: str) -> None:
    print(f"\n{SEP}\n{msg}\n{SEP}", flush=True)


# --------------------------------------------------------------------------
# Stage 0 - load
# --------------------------------------------------------------------------
def stage_load():
    banner("STAGE 0  load dataset")
    ds = load_dataset(DATASET_NAME)
    for split, d in ds.items():
        print(f"  split={split:<6} rows={d.num_rows:>6}  columns={d.column_names}",
              flush=True)
    return ds


# --------------------------------------------------------------------------
# Stage 1 - flatten to one DataFrame with a split column
# --------------------------------------------------------------------------
def stage_flatten(ds) -> pd.DataFrame:
    banner("STAGE 1  flatten splits -> single DataFrame")
    frames = []
    for split, d in ds.items():
        df = d.to_pandas()
        # Keep only known fields; warn loudly if schema differs from expectation.
        missing = [c for c in FIELDS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Split {split!r} is missing expected fields {missing}; "
                f"got columns {list(df.columns)}"
            )
        df = df[FIELDS].copy()
        df["split"] = split
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    # Normalise dtype to str (None preserved as NaN for null detection below).
    print(f"  combined shape = {full.shape}")
    print(f"  split counts   = {full['split'].value_counts().to_dict()}")
    print("  head (truncated):")
    with pd.option_context("display.max_colwidth", 60):
        print(full.head(3).to_string())
    return full


# --------------------------------------------------------------------------
# Stage 2 - profiling  (the inspectable quality step)
# --------------------------------------------------------------------------
def _char_len(series: pd.Series) -> pd.Series:
    # NaN -> length 0 is wrong for stats; compute only on non-null strings.
    return series.dropna().astype(str).str.len()


def _len_stats(series: pd.Series) -> dict:
    lens = _char_len(series)
    if lens.empty:
        return dict(min=0, median=0, mean=0.0, max=0, count=0)
    return dict(
        min=int(lens.min()),
        median=float(lens.median()),
        mean=float(lens.mean()),
        max=int(lens.max()),
        count=int(lens.count()),
    )


def _null_empty(series: pd.Series) -> dict:
    nulls = int(series.isna().sum())
    # empty = present but blank / whitespace-only
    nonnull = series.dropna().astype(str)
    empty = int((nonnull.str.strip() == "").sum())
    return dict(nulls=nulls, empty=empty)


def stage_profile(full: pd.DataFrame) -> dict:
    banner("STAGE 2  profile / quality audit")
    profile = {}

    profile["total_rows"] = int(len(full))
    profile["split_counts"] = full["split"].value_counts().to_dict()
    print(f"  total rows = {profile['total_rows']}")
    print(f"  split counts = {profile['split_counts']}")

    profile["len_stats"] = {}
    profile["null_empty"] = {}
    for f in FIELDS:
        st = _len_stats(full[f])
        ne = _null_empty(full[f])
        profile["len_stats"][f] = st
        profile["null_empty"][f] = ne
        print(f"  [{f:<8}] len min/median/mean/max = "
              f"{st['min']}/{st['median']:.1f}/{st['mean']:.1f}/{st['max']}"
              f"   nulls={ne['nulls']} empty={ne['empty']}")

    # Duplicate analysis
    dup_ctx_rows = int(full["context"].duplicated(keep=False).sum())
    dup_q_rows = int(full["question"].duplicated(keep=False).sum())
    n_unique_ctx = int(full["context"].nunique(dropna=True))
    n_unique_q = int(full["question"].nunique(dropna=True))
    # Exact-duplicate count = rows that are repeats of an earlier identical value
    dup_ctx_extra = int(full["context"].duplicated(keep="first").sum())
    dup_q_extra = int(full["question"].duplicated(keep="first").sum())

    profile["dup"] = dict(
        context_rows_in_dup_groups=dup_ctx_rows,
        context_duplicate_extra=dup_ctx_extra,
        unique_contexts=n_unique_ctx,
        question_rows_in_dup_groups=dup_q_rows,
        question_duplicate_extra=dup_q_extra,
        unique_questions=n_unique_q,
    )
    print(f"  contexts : unique={n_unique_ctx}  "
          f"rows_in_dup_groups={dup_ctx_rows}  redundant_copies={dup_ctx_extra}")
    print(f"  questions: unique={n_unique_q}  "
          f"rows_in_dup_groups={dup_q_rows}  redundant_copies={dup_q_extra}")

    # Cross-check: do duplicate contexts have inconsistent answers? (informational)
    profile["context_to_nanswers"] = None
    grp = full.dropna(subset=["context"]).groupby("context")["answer"].nunique()
    multi = grp[grp > 1]
    profile["context_to_nanswers"] = dict(
        contexts_with_multiple_distinct_answers=int((grp > 1).sum()),
        max_distinct_answers_per_context=int(grp.max()) if len(grp) else 0,
    )
    print(f"  contexts mapping to >1 distinct answer = "
          f"{profile['context_to_nanswers']['contexts_with_multiple_distinct_answers']}"
          f" (max {profile['context_to_nanswers']['max_distinct_answers_per_context']})")

    return profile


# --------------------------------------------------------------------------
# Stage 3 - build corpus (unique contexts, stable ids)
# --------------------------------------------------------------------------
def stage_build_corpus(full: pd.DataFrame):
    banner("STAGE 3  build corpus.parquet (unique contexts)")
    # Drop null contexts from the corpus but track how many were dropped.
    valid = full.dropna(subset=["context"]).copy()
    dropped = len(full) - len(valid)
    if dropped:
        print(f"  WARNING: {dropped} rows had null context and are excluded "
              f"from the corpus.")

    # First-occurrence order gives deterministic ids.
    unique_ctx = (
        valid.drop_duplicates(subset=["context"], keep="first")
        .reset_index(drop=True)
        .loc[:, ["context"]]
    )
    width = max(6, len(str(len(unique_ctx))))
    unique_ctx.insert(
        0, "context_id",
        ["ctx_" + str(i + 1).zfill(width) for i in range(len(unique_ctx))],
    )

    # Mapping context-text -> context_id, used to label the eval set.
    ctx_to_id = dict(zip(unique_ctx["context"], unique_ctx["context_id"]))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    unique_ctx.to_parquet(CORPUS_PATH, index=False)
    size = CORPUS_PATH.stat().st_size
    print(f"  unique contexts = {len(unique_ctx)}")
    print(f"  wrote {CORPUS_PATH}  ({size/1e6:.2f} MB)")
    print("  first ids:", unique_ctx["context_id"].head(3).tolist(),
          "... last:", unique_ctx["context_id"].iloc[-1])
    return unique_ctx, ctx_to_id


# --------------------------------------------------------------------------
# Stage 4 - build eval set (qrels + gold answer)
# --------------------------------------------------------------------------
def stage_build_eval(full: pd.DataFrame, ctx_to_id: dict):
    banner("STAGE 4  build eval.parquet (qrels + gold answer)")
    ev = full.copy()
    ev["gold_context_id"] = ev["context"].map(ctx_to_id)

    unmapped = int(ev["gold_context_id"].isna().sum())
    if unmapped:
        print(f"  WARNING: {unmapped} eval rows have no gold_context_id "
              f"(null context). They are kept with gold_context_id=NaN; flag downstream.")

    ev = ev.loc[:, ["question", "answer", "gold_context_id", "split"]].copy()
    ev.to_parquet(EVAL_PATH, index=False)
    size = EVAL_PATH.stat().st_size
    print(f"  eval rows = {len(ev)}")
    print(f"  wrote {EVAL_PATH}  ({size/1e6:.2f} MB)")
    print("  head (truncated):")
    with pd.option_context("display.max_colwidth", 50):
        print(ev.head(3).to_string())

    # Validation: every non-null context must map to a valid corpus id.
    valid_ids = set(ctx_to_id.values())
    bad = int((~ev["gold_context_id"].dropna().isin(valid_ids)).sum())
    print(f"  validation: eval rows with invalid gold_context_id = {bad}")
    return ev, unmapped, bad


# --------------------------------------------------------------------------
# Stage 5 - DATA_QUALITY.md
# --------------------------------------------------------------------------
def stage_report(full, profile, unique_ctx, ev, unmapped, bad_ids):
    banner("STAGE 5  write DATA_QUALITY.md")

    def trunc(s, n=300):
        s = "" if s is None else str(s)
        s = s.replace("\n", " ").replace("\r", " ").strip()
        return s if len(s) <= n else s[:n] + " ...[truncated]"

    examples = full.head(3)
    ex_blocks = []
    for i, (_, r) in enumerate(examples.iterrows(), 1):
        ex_blocks.append(
            f"### Example {i} (split={r['split']})\n\n"
            f"- **question** ({len(str(r['question']))} chars): {trunc(r['question'])}\n"
            f"- **answer** ({len(str(r['answer']))} chars): {trunc(r['answer'])}\n"
            f"- **context** ({len(str(r['context']))} chars): {trunc(r['context'])}\n"
        )
    examples_md = "\n".join(ex_blocks)

    ls = profile["len_stats"]
    ne = profile["null_empty"]
    dp = profile["dup"]
    ca = profile["context_to_nanswers"]

    def lenrow(f):
        s = ls[f]
        return (f"| {f} | {s['min']} | {s['median']:.0f} | {s['mean']:.1f} | "
                f"{s['max']} | {ne[f]['nulls']} | {ne[f]['empty']} |")

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    report = f"""# DATA QUALITY REPORT - rag-dataset-12000

Generated: {generated}
Source: HuggingFace `neural-bridge/rag-dataset-12000`
Pipeline: `scripts/01_prepare_data.py` (self-contained, re-runnable)

## 1. Row counts

- Total rows loaded: **{profile['total_rows']}**
- Per split: {profile['split_counts']}

## 2. Field length distributions (characters) and null/empty

| field | min | median | mean | max | nulls | empty |
|-------|-----|--------|------|-----|-------|-------|
{lenrow('context')}
{lenrow('question')}
{lenrow('answer')}

`empty` = present but whitespace-only. `nulls` = missing values.

## 3. Duplicate / uniqueness analysis

- Unique contexts: **{dp['unique_contexts']}** (out of {profile['total_rows']} rows)
- Context rows belonging to a duplicate group: {dp['context_rows_in_dup_groups']}
- Redundant context copies (rows beyond first occurrence): {dp['context_duplicate_extra']}
- Unique questions: {dp['unique_questions']}
- Question rows belonging to a duplicate group: {dp['question_rows_in_dup_groups']}
- Redundant question copies: {dp['question_duplicate_extra']}

Cross-check: contexts mapping to more than one distinct gold answer:
**{ca['contexts_with_multiple_distinct_answers']}**
(max distinct answers for a single context: {ca['max_distinct_answers_per_context']}).
This is expected if the same passage is paired with different question/answer rows.

## 4. Built artifacts

- `data/processed/corpus.parquet` - retrievable corpus, one row per UNIQUE context.
  Columns: `context_id` (stable string, e.g. `ctx_000001`), `context` (text).
  Rows: **{len(unique_ctx)}**.
- `data/processed/eval.parquet` - evaluation set / qrels.
  Columns: `question`, `answer` (gold), `gold_context_id` (corpus id of the
  source context), `split` (train/test).
  Rows: **{len(ev)}**.

Validation:
- eval rows with no gold_context_id (null context): {unmapped}
- eval rows with an invalid gold_context_id: {bad_ids}

## 5. Example rows (truncated)

{examples_md}

## 6. License

Dataset license: **Apache 2.0**. The underlying context passages are drawn from
the Falcon RefinedWeb corpus; downstream use must follow Falcon RefinedWeb /
ODC-By terms (attribution, no additional restrictions). Keep this notice with any
redistribution of the corpus.

## 7. Caveats / limitations (honest)

- **Null question/answer rows.** {ne['question']['nulls']} rows have a NULL
  `question` and {ne['answer']['nulls']} rows have a NULL `answer` (same rows: these are the
  same rows: question AND answer both missing, but the context is present and
  valid). These rows survive into `eval.parquet` with `gold_context_id` set and
  `question`/`answer` = NaN. Downstream evaluation MUST drop rows where `question`
  is null before scoring; their contexts still legitimately belong in the corpus.
  This contradicts the "no nulls" assumption in the supplied dataset facts.
- **Single-relevant qrels.** Each question maps to exactly ONE gold context
  (`gold_context_id`). Retrieval metrics computed from this (Recall@k, MRR, nDCG)
  treat all other contexts as non-relevant, even though other passages in the
  corpus may also answer the question. Recall/precision are therefore a LOWER
  bound on true retrieval quality. Do not over-interpret absolute scores.
- **Corpus = the set of contexts only.** There is no external document pool; the
  retrievable universe is exactly the {len(unique_ctx)} unique contexts.
- **Duplicate contexts collapsed.** {dp['context_duplicate_extra']} redundant
  context copies were merged into shared ids. If two questions share an identical
  context, they share a gold_context_id (correct for retrieval, but means some
  contexts are "relevant" to multiple questions).
- **Answers are model/gold text, not span offsets.** The `answer` field is free
  text, not a character span into the context; exact-match against the context is
  not guaranteed.
- Length ranges observed here should be compared against the documented dataset
  facts (context 271-9160, question 17-322, answer 2-2410 chars). Any deviation is
  reported in the tables above and should be treated as the source of truth for
  THIS download.
"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"  wrote {REPORT_PATH}  ({REPORT_PATH.stat().st_size} bytes)")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ds = stage_load()
    full = stage_flatten(ds)
    profile = stage_profile(full)
    unique_ctx, ctx_to_id = stage_build_corpus(full)
    ev, unmapped, bad_ids = stage_build_eval(full, ctx_to_id)
    stage_report(full, profile, unique_ctx, ev, unmapped, bad_ids)

    banner("SUMMARY")
    print(f"  source            : {DATASET_NAME}")
    print(f"  total rows        : {profile['total_rows']} "
          f"({profile['split_counts']})")
    print(f"  unique contexts   : {profile['dup']['unique_contexts']}")
    print(f"  corpus.parquet    : {CORPUS_PATH}  "
          f"({CORPUS_PATH.stat().st_size/1e6:.2f} MB, {len(unique_ctx)} rows)")
    print(f"  eval.parquet      : {EVAL_PATH}  "
          f"({EVAL_PATH.stat().st_size/1e6:.2f} MB, {len(ev)} rows)")
    print(f"  report            : {REPORT_PATH}")
    print(f"  eval rows unmapped: {unmapped}   invalid gold ids: {bad_ids}")
    print("  DONE.")


if __name__ == "__main__":
    main()
