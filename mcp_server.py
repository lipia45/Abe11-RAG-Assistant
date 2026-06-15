"""
MCP server - exposes the RAG retriever as a Model Context Protocol tool.

Any MCP client (Claude Desktop, Claude Agent SDK, etc.) can connect and call
`search_knowledge_base` to retrieve cited passages from the private collection.

Run:  py mcp_server.py        (stdio transport)
Deps: pip install "mcp[cli]"   (FastMCP)

Reuses the Phase A winning retrieval index (chunk900_150, cached embeddings) via
the Retriever in agent_rag.py - no re-embedding.
"""
from mcp.server.fastmcp import FastMCP
from agent_rag import Retriever, TOPK

mcp = FastMCP("rag-knowledge-base")
_retriever = Retriever()


@mcp.tool()
def search_knowledge_base(query: str, k: int = TOPK) -> list[dict]:
    """Search the private document collection for passages relevant to `query`.

    Returns up to `k` passages, each as {"id": "ctx_000123", "text": "..."}.
    Cite the returned ids in any answer built from these passages.
    """
    return _retriever.search(query, k)


if __name__ == "__main__":
    mcp.run()  # stdio transport
