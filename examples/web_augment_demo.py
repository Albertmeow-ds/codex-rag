"""End-to-end web augmentation demo.

Tries a real live search first. If the sandbox has no network, it falls back to a
captured page so the whole web -> chunk -> index -> retrieve -> answer chain still runs.
"""

import sys

sys.path.insert(0, ".")

from pathlib import Path

from rag.config import RAGConfig
from rag.pipeline import RAGPipeline
from rag.retrieve.web import WebSearchRetriever
from rag.types import Document

QUERY = "graph rag 什么时候比向量检索更合适"

CAPTURED_PAGE = """# GraphRAG vs vector retrieval

GraphRAG extracts entities and relations from the corpus and builds a graph, then
retrieves by community detection and multi-hop traversal. It wins on aggregative
questions that span many documents, for example "what trend do these reports jointly
indicate". For single-hop factual questions plain vector retrieval is cheaper and
usually better, because graph construction cost is high and the graph adds no signal.

## When vector retrieval is enough

If the answer lives in one passage, dense retrieval plus reranking is the right tool.
GraphRAG pays off when the answer requires joining evidence across documents.
"""


def main() -> int:
    db = Path("data/web_demo.db")
    if db.exists():
        db.unlink()
    config = RAGConfig.from_file("config.yaml")
    config.db_path = str(db)
    config.llm = "none"
    pipeline = RAGPipeline(config)
    pipeline.index_paths(["examples/corpus"])

    web = WebSearchRetriever(max_results=3)
    live = web.available()
    print(f"live network available: {live}")

    if live:
        documents = web.search_documents(QUERY, max_results=3)
        print(f"fetched {len(documents)} live pages")
    else:
        print("network blocked -> using a captured page so the chain is still exercised")
        documents = [
            Document(
                doc_id="web:captured-graphrag",
                text=CAPTURED_PAGE,
                source="https://example.com/graphrag-vs-vector",
                title="GraphRAG vs vector retrieval",
                metadata={"origin": "web"},
            )
        ]

    before = pipeline.store.counts()["documents"]
    pipeline.augment(documents)
    after = pipeline.store.counts()["documents"]
    print(f"indexed documents {before} -> {after}")

    english_query = "when is graphrag better than vector retrieval for aggregative questions"
    hits = pipeline.retrieve(english_query, top_k=5)
    print("\nretrieval after augmentation")
    for unit in hits.units:
        origin = unit.chunk.metadata.get("origin", "local")
        print(f"  {unit.rank}. [{origin}] {unit.chunk.doc_id}  score={unit.final_score:.4f}")

    web_hits = [u for u in hits.units if u.chunk.metadata.get("origin") == "web"]
    print(f"\nweb-sourced candidates in top-5: {len(web_hits)}")

    cross_lingual = pipeline.retrieve(QUERY, top_k=5)
    print(f"\nsame question in Chinese -> web-sourced candidates: {len([u for u in cross_lingual.units if u.chunk.metadata.get('origin') == 'web'])}")
    print("(expected: 0. BM25 and hashed embeddings have no cross-lingual transfer; a real multilingual embedding model fixes this)")

    answer = pipeline.answer(english_query, agentic=False)
    print("\nanswer")
    print(answer.text)
    print("\ncitations")
    for citation in answer.citations:
        print(f"  [{citation.index}] {citation.doc_id}")
    print(f"\ngroundedness={answer.groundedness:.3f}")
    print("trace: " + answer.trace.summary())

    pipeline.close()
    return 0 if web_hits else 1


if __name__ == "__main__":
    raise SystemExit(main())