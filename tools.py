"""
tools.py
--------
Defines tools used by the Alfred RAG agent.

Tools:
    GuestInfoRetrieverTool  – local BM25 retrieval over the invitees KB
"""

from smolagents import Tool
from retriever import retrieve, DOCUMENTS


class GuestInfoRetrieverTool(Tool):
    """
    Local knowledge-base retrieval tool for the gala invitees dataset.

    Uses the from-scratch BM25 retriever (see retriever.py) to find the
    most relevant guest records for a natural-language query.

    The agent should call this tool whenever a question is about a specific
    guest, their background, their relation to the host, or their email.
    """

    name = "guest_info_retriever"
    description = (
        "Retrieves detailed information about gala guests from the local knowledge base. "
        "Input should be the guest's name, their relation to the host, or keywords "
        "describing them (e.g. 'physicist', 'best friend', 'university days'). "
        "Returns the top-3 matching guest records including name, relation, "
        "description, and email. "
        "ALWAYS call this tool before answering any question about a specific guest."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": (
                "A search query containing the guest's name, relation to the host, "
                "or descriptive keywords about the guest."
            ),
        }
    }
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, query: str) -> str:
        """Run BM25 retrieval and return formatted top-3 results."""
        results = retrieve(query, k=3)

        if not results:
            return "No matching guest information found in the knowledge base."

        sections = []
        for i, doc in enumerate(results, 1):
            sections.append(f"[Result {i}]\n{doc.page_content}")

        return "\n\n".join(sections)


# ── Instantiate tools for import by app.py ────────────────────────────────────
guest_info_tool = GuestInfoRetrieverTool()


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "Tell me about Lady Ada Lovelace",
        "Who is the best friend of the host?",
        "Marie Curie background",
    ]
    for q in test_queries:
        print(f"\nQuery : {q}")
        print("-" * 50)
        print(guest_info_tool.forward(q))
