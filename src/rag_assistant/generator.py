"""Answer generation with Anthropic Claude (Phase 4 stub).

When implemented, :func:`answer` will build a prompt from the user query and the
retrieved contexts, call Claude via the ``anthropic`` SDK, and return a grounded,
*cited* answer: each claim references the context chunk(s) it came from (e.g. ``[1]``,
``[2]`` mapped to the contexts' ids), and the model is instructed to say it does not
know when the contexts do not support an answer.

The API key is read from the environment (``ANTHROPIC_API_KEY``) and is never
hardcoded. This stub does NOT call the API yet.
"""

from __future__ import annotations

from typing import List, Sequence

from .config import settings
from .retriever import Hit


def format_contexts(contexts: Sequence[Hit]) -> str:
    """Render retrieved hits as a numbered context block for the prompt.

    Args:
        contexts: Retrieved hits, in rank order.

    Returns:
        A string like::

            [1] (id=...) <chunk text>
            [2] (id=...) <chunk text>
    """
    lines: List[str] = []
    for i, h in enumerate(contexts, start=1):
        lines.append(f"[{i}] (id={h.id}) {h.text}")
    return "\n\n".join(lines)


def build_prompt(query: str, contexts: Sequence[Hit]) -> str:
    """Build the user-message prompt instructing a cited, grounded answer."""
    context_block = format_contexts(contexts)
    return (
        "Answer the question using ONLY the numbered context below. "
        "Cite the supporting context with its bracket number, e.g. [1]. "
        "If the context does not contain the answer, say you don't know.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        "Answer (with citations):"
    )


def answer(query: str, contexts: Sequence[Hit]) -> str:
    """Generate a cited answer from the query and retrieved contexts.

    TODO (Phase 4): implement the Claude call. Sketch::

        from anthropic import Anthropic
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        client = Anthropic(api_key=settings.anthropic_api_key)
        resp = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": build_prompt(query, contexts)}],
        )
        return resp.content[0].text

    Args:
        query: The user question.
        contexts: Retrieved hits to ground the answer.

    Returns:
        The generated answer string with inline citations.

    Raises:
        NotImplementedError: Always, until Phase 4 wires in the Claude call.
    """
    raise NotImplementedError(
        "generator.answer() is a Phase 4 stub. "
        "Prompt is ready via build_prompt(); wire the anthropic SDK call here."
    )
