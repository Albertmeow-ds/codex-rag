"""Real live web retrieval: search -> fetch -> index -> retrieve -> answer -> compare.

Run:  python examples/live_web_search.py
"""

import sys
import time

sys.path.insert(0, ".")

from pathlib import Path

from rag.config import RAGConfig
from rag.eval import evaluate_retrieval, load_dataset
from rag.pipeline import RAGPipeline
from rag.retrieve.web import WebSearchRetriever

ARXIV_QUERIES = [
    'abs:"retrieval-augmented generation" AND cat:cs.IR',
    'abs:"agentic retrieval" OR abs:"corrective RAG"',
    'abs:"reranking" AND abs:"retrieval-augmented generation"',
    'ti:"chunking" AND abs:"retrieval-augmented generation"',
]
BING_QUERIES = [
    "GraphRAG multi-hop retrieval",
    "agentic RAG framework",
    "RAG evaluation faithfulness metrics",
]


def main() -> int:
    db = Path("data/live_web.db")
    if db.exists():
        db.unlink()
    config = RAGConfig.from_file("config.yaml")
    config.db_path = str(db)
    config.llm = "none"

    pipeline = RAGPipeline(config)
    pipeline.index_paths(["examples/corpus"])
    cases = load_dataset("examples/eval.jsonl")

    baseline = evaluate_retrieval(cases, lambda q: pipeline.retrieve(q).chunk_ids, k=5)
    print("local-only corpus baseline")
    for key, value in baseline.items():
        print(f"  {key} = {value}")

    web = WebSearchRetriever(max_results=6, fetch_pages=True)
    print(f"\nnetwork available: {web.available()}")

    documents = []
    started = time.time()
    for query in ARXIV_QUERIES:
        web.backend = "arxiv"
        try:
            fetched = web.search_documents(query, max_results=5)
        except Exception as exc:
            print(f"  arxiv failed for {query!r}: {type(exc).__name__}")
            continue
        print(f"  arxiv {query[:52]:<52} -> {len(fetched)} docs")
        documents.extend(fetched)

    for query in BING_QUERIES:
        web.backend = "bing"
        try:
            fetched = web.search_documents(query, max_results=4)
        except Exception as exc:
            print(f"  bing failed for {query!r}: {type(exc).__name__}")
            continue
        print(f"  bing  {query[:52]:<52} -> {len(fetched)} docs")
        documents.extend(fetched)

    fetched_pages = [d for d in documents if len(d.text) > 400]
    print(f"\nsearched {len(ARXIV_QUERIES) + len(BING_QUERIES)} queries in {time.time() - started:.1f}s")
    print(f"documents fetched: {len(documents)} (usable >400 chars: {len(fetched_pages)})")
    for document in documents[:6]:
        print(f"  - {document.title[:70]}")
        print(f"    {document.source[:90]}  chars={len(document.text)}")

    before = pipeline.store.counts()
    pipeline.augment(documents)
    after = pipeline.store.counts()
    print(f"\nindex: docs {before['documents']} -> {after['documents']}, chunks {before['chunks']} -> {after['chunks']}, vectors {before['vectors']} -> {after['vectors']}")

    print("\nretrieval after web augmentation (same local questions)")
    augmented = evaluate_retrieval(cases, lambda q: pipeline.retrieve(q).chunk_ids, k=5)
    for key, value in augmented.items():
        delta = value - baseline[key]
        print(f"  {key} = {value}  ({delta:+.4f})")

    web_only_questions = [
        "what is microsoft graphrag and how does it build the graph",
        "how do recent agentic RAG systems decide when to retrieve",
        "what metrics do recent RAG surveys use to judge faithfulness",
    ]
    print("\nquestions the local corpus cannot answer")
    for question in web_only_questions:
        hits = pipeline.retrieve(question, top_k=5)
        web_hits = [u for u in hits.units if u.chunk.metadata.get("origin") == "web"]
        answer = pipeline.answer(question, agentic=False)
        print(f"\nQ: {question}")
        print(f"  web-sourced candidates in top-5: {len(web_hits)} / {len(hits.units)}")
        print("  answer:")
        for line in answer.text.split("\n")[:4]:
            print(f"    {line[:150]}")
        print(f"  groundedness={answer.groundedness:.3f}")

    print("\nsource-filtered retrieval (web only)")
    web_hits = pipeline.retrieve("graphrag entity graph community detection", top_k=5, filters={"metadata": {"origin": "web"}})
    for unit in web_hits.units:
        print(f"  [{unit.chunk.metadata.get('origin')}] {unit.chunk.doc_id[:60]}  {unit.final_score:.3f}")

    pipeline.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())