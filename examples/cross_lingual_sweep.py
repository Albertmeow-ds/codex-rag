"""Cross-lingual sweep: does a Chinese query reach English web documents?

Compares two ways of turning arXiv hits into Documents:
  fetched  -> download the abs page and strip HTML (nav-heavy)
  abstract -> use the Atom summary / abstract blockquote (clean prose)

Run:  python examples/cross_lingual_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, ".")

from rag.config import RAGConfig
from rag.eval import evaluate_retrieval, load_dataset
from rag.pipeline import RAGPipeline
from rag.retrieve.web import WebSearchRetriever

WEB_QUERY = 'abs:"retrieval-augmented generation" AND cat:cs.IR'

QUERIES = [
    ("local", "重排和后期交互对检索质量有什么影响"),
    ("local", "graphrag 什么时候比向量检索更合适"),
    ("local", "近期论文如何评估检索增强生成系统"),
    ("web-only", "校园 AI 导师在硬件和软件之间如何权衡"),
    ("web-only", "多跳问答中小块和大块应该怎么安排才能省成本"),
]

VARIANTS = [
    ("weighted 1.2:0.8", ["bm25", "dense"], "weighted", {"bm25": 1.2, "dense": 0.8, "late": 0.5}),
    ("weighted 1:1", ["bm25", "dense"], "weighted", {"bm25": 1.0, "dense": 1.0, "late": 0.5}),
    ("weighted 0.5:1.5", ["bm25", "dense"], "weighted", {"bm25": 0.5, "dense": 1.5, "late": 0.5}),
    ("dense only", ["dense"], "weighted", {"dense": 1.0}),
    ("rrf 1:1", ["bm25", "dense"], "rrf", {"bm25": 1.0, "dense": 1.0, "late": 0.5}),
]


def build_web_documents(mode: str):
    if mode == "abstract":
        web = WebSearchRetriever(max_results=4, backend="arxiv", prefer_snippet=True)
    else:
        web = WebSearchRetriever(max_results=4, backend="arxiv", prefer_snippet=False)
    return web.search_documents(WEB_QUERY, max_results=4)


def run_variants(web_documents, mode: str, cases) -> None:
    for index, (label, channels, fusion, weights) in enumerate(VARIANTS):
        db = Path("data") / f"xling_{mode}_{index}.db"
        db.unlink(missing_ok=True)
        config = RAGConfig.from_file("config.bge-m3.yaml")
        config.db_path = str(db)
        config.llm = "none"
        config.embedder = "ollama:bge-m3"
        config.reranker = "none"
        config.retrieval.channels = tuple(channels)
        config.retrieval.fusion = fusion
        config.retrieval.weights = weights
        pipeline = RAGPipeline(config)
        pipeline.index_paths(["examples/corpus"])
        pipeline.augment(web_documents)
        metrics = evaluate_retrieval(cases, lambda question: pipeline.retrieve(question).chunk_ids, k=5)
        web_counts = []
        for kind, question in QUERIES:
            hits = pipeline.retrieve(question, top_k=5)
            web_units = [unit for unit in hits.units if unit.chunk.metadata.get("origin") == "web"]
            web_counts.append(len(web_units))
            if kind == "web-only":
                best_web = max((unit.final_score for unit in web_units), default=float("nan"))
                best_local = max(
                    (unit.final_score for unit in hits.units if unit.chunk.metadata.get("origin") != "web"),
                    default=float("nan"),
                )
                print(f"      [{kind}] {question[:28]} web_best={best_web:.4f} local_best={best_local:.4f}")
        print(
            f"  {label:<18} local recall={metrics['recall@5']:.4f} ndcg={metrics['ndcg@5']:.4f} "
            f"| web hits in top5: {web_counts}"
        )
        pipeline.close()
        db.unlink(missing_ok=True)


def main() -> int:
    cases = load_dataset("examples/eval.jsonl")
    for mode in ("fetched", "abstract"):
        documents = build_web_documents(mode)
        print(f"\n=== web documents via {mode}: {len(documents)} ===")
        for document in documents[:2]:
            print(f"  [{len(document.text)} chars] {document.title[:60]}")
            print(f"     {document.text[:160]}")
        run_variants(documents, mode, cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
