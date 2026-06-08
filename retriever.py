import math
import re
from collections import Counter
import datasets

class Document:
    """Simple document container mirroring langchain_core Document interface."""

    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Document(metadata={self.metadata}, content_snippet={self.page_content[:80]!r})"


#BM25

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


class BM25:
    """
    Okapi BM25 retriever implemented from scratch.

    Parameters
    ----------
    k1 : float  Controls term-frequency saturation (default 1.5)
    b  : float  Controls document-length normalization (default 0.75)
    """

    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b

        # Tokenize all documents
        self.tokenized_docs: list[list[str]] = [
            _tokenize(doc.page_content) for doc in documents
        ]

        self.N = len(self.tokenized_docs)
        self.avgdl = sum(len(d) for d in self.tokenized_docs) / self.N if self.N else 1

        # Term-frequency per document: list of Counter
        self.tf: list[Counter] = [Counter(td) for td in self.tokenized_docs]

        # Document frequency: how many docs contain each term
        self.df: Counter = Counter()
        for td in self.tokenized_docs:
            for term in set(td):
                self.df[term] += 1

    def _idf(self, term: str) -> float:
        """Smoothed IDF so unseen terms still get a small positive score."""
        n_t = self.df.get(term, 0)
        return math.log((self.N - n_t + 0.5) / (n_t + 0.5) + 1)

    def _score(self, query_terms: list[str], doc_idx: int) -> float:
        """Compute BM25 score for a single document."""
        score = 0.0
        dl = len(self.tokenized_docs[doc_idx])
        tf_doc = self.tf[doc_idx]
        for term in query_terms:
            f = tf_doc.get(term, 0)
            idf = self._idf(term)
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * (numerator / denominator)
        return score

    def retrieve(self, query: str, k: int = 3) -> list[tuple[Document, float]]:
        """
        Return the top-k documents most relevant to *query*.

        Returns
        -------
        List of (Document, score) tuples, sorted descending by score.
        """
        query_terms = _tokenize(query)
        scored = [
            (self.documents[i], self._score(query_terms, i))
            for i in range(self.N)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

def load_invitees_dataset() -> list[Document]:
    """
    Load agents-course/unit3-invitees from HuggingFace and convert each
    record to a Document with structured page_content.
    """
    guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

    docs = []
    for guest in guest_dataset:
        content = "\n".join([
            f"Name: {guest['name']}",
            f"Relation: {guest['relation']}",
            f"Description: {guest['description']}",
            f"Email: {guest['email']}",
        ])
        docs.append(Document(
            page_content=content,
            metadata={"name": guest["name"], "email": guest["email"]}
        ))

    return docs


# Build the retriever once at module load so tools.py can import it directly.
print("Loading invitees dataset …")
DOCUMENTS = load_invitees_dataset()
print(f"  → {len(DOCUMENTS)} documents loaded.")

BM25_RETRIEVER = BM25(DOCUMENTS, k1=1.5, b=0.75)
print("BM25 index built.\n")


def retrieve(query: str, k: int = 3) -> list[Document]:
    """
    Retrieve the top-k most relevant Documents for *query*.
    Returns only the Document objects (scores are discarded).
    """
    results = BM25_RETRIEVER.retrieve(query, k=k)
    return [doc for doc, _ in results]


def retrieve_with_scores(query: str, k: int = 3) -> list[tuple[Document, float]]:
    """Same as retrieve() but also returns BM25 scores."""
    return BM25_RETRIEVER.retrieve(query, k=k)


#Test
if __name__ == "__main__":
    manual_queries = [
        "Ada Lovelace",
        "old friend from university",
        "physicist radioactivity scientist",
        "gmail email contact",
        "mathematician programmer computing",
    ]

    for q in manual_queries:
        results = retrieve_with_scores(q, k=3)
        print(f"\n{'='*60}")
        print(f"QUERY : {q}")
        print(f"{'='*60}")
        for rank, (doc, score) in enumerate(results, 1):
            print(f"\n  -- Result {rank} (BM25 score: {score:.4f}) --")
            print(f"  {doc.page_content.replace(chr(10), '  |  ')}")
        best = results[0][0].metadata["name"]
        print(f"\n  >>> Best match: {best}")
