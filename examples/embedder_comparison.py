"""hashed vs lsa vs bge-m3 on the same corpus, including cross-lingual retrieval."""

import sys
import time

sys.path.insert(0, ".")

from pathlib import Path

from rag.config import RAGConfig
from rag.eval import evaluate_retrieval, load_dataset
from rag.pipeline import RAGPipeline
from rag.retrieve.web import WebSearchRetriever

CROSS_LINGUAL = [
    ("中文问题 -> 英文文档", "graphrag 什么时候比向量检索更合适", "graphrag"),
    ("中文问题 -> 英文文档", "重排和后期交互对检索质量有什么影响", "late interaction"),
    ("英文问题 -> 中文文档", "how does hierarchical chunking work", "chunking"),
]


def build(embedder_name: str, db_name: str, reranker: str) -> RAGPipeline:
    db = Path("data") / db_name
    if db.exists():
        db.unlink()
    config = RAGConfig.from_file("config.yaml")
    config.db_path = str(db)
    config.llm = "none"
    config.embedder = embedder_name
    config.reranker = reranker
    pipeline = RAGPipeline(config)
    pipeline.index_paths(["examples/corpus"])
    return pipeline


WEIGHTS = [
    (1.2, 0.8),
    (1.0, 1.0),
    (0.8, 1.2),
    (0.5, 1.5),
]


def weight_sweep(cases) -> None:
    """Fusion weights are not transferable between embedders; measure them per model."""

    print("\nfusion weight sweep (local corpus only, channels bm25+dense)")
    for embedder_name in ("hashed", "ollama:bge-m3"):
        for reranker in ("none", "heuristic"):
            row = []
            for bm25_weight, dense_weight in WEIGHTS:
                db = Path("data") / f"cmp_w_{embedder_name.replace(':', '_')}_{reranker}_{bm25_weight}_{dense_weight}.db"
                db.unlink(missing_ok=True)
                config = RAGConfig.from_file("config.yaml")
                config.db_path = str(db)
                config.llm = "none"
                config.embedder = embedder_name
                config.reranker = reranker
                config.retrieval.weights = {"bm25": bm25_weight, "dense": dense_weight, "late": 0.5}
                pipeline = RAGPipeline(config)
                try:
                    pipeline.index_paths(["examples/corpus"])
                    metrics = evaluate_retrieval(cases, lambda q: pipeline.retrieve(q).chunk_ids, k=5)
                    row.append((bm25_weight, dense_weight, metrics))
                finally:
                    pipeline.close()
                    db.unlink(missing_ok=True)
                row[-1] = (bm25_weight, dense_weight, metrics)
            best = max(row, key=lambda item: item[2]["ndcg@5"])
            cells = "  ".join(
                f"{w0}:{w1} r={m['recall@5']:.3f}/n={m['ndcg@5']:.3f}" for w0, w1, m in row
            )
            print(f"  {embedder_name:<14} + {reranker:<9} -> best {best[0]}:{best[1]}   {cells}")


def main() -> int:
    cases = load_dataset("examples/eval.jsonl")
    web = WebSearchRetriever(max_results=4, fetch_pages=True, backend="arxiv")
    try:
        web_docs = web.search_documents('abs:"retrieval-augmented generation" AND cat:cs.IR', max_results=4)
    except Exception as exc:
        print("web fetch failed:", exc)
        web_docs = []
    print(f"web documents: {len(web_docs)}")

    for embedder_name in ("hashed", "lsa", "ollama:bge-m3"):
        for reranker in ("none", "heuristic", "late"):
            label = f"{embedder_name:<14} + {reranker:<9}"
            pipeline = build(embedder_name, f"cmp_{embedder_name.replace(':','_')}_{reranker}.db", reranker)
            started = time.time()
            metrics = evaluate_retrieval(cases, lambda q: pipeline.retrieve(q).chunk_ids, k=5)
            index_time = time.time() - started
            print(
                f"{label} recall={metrics['recall@5']:.4f} prec={metrics['precision@5']:.4f} "
                f"mrr={metrics['mrr@5']:.4f} ndcg={metrics['ndcg@5']:.4f}  ({index_time:.1f}s)"
            )
            pipeline.close()

    weight_sweep(cases)

    print("\ncross-lingual retrieval (local corpus + live English web pages)")
    for embedder_name in ("hashed", "ollama:bge-m3"):
        pipeline = build(embedder_name, f"cmp_x_{embedder_name.replace(':','_')}.db", "heuristic")
        if web_docs:
            pipeline.augment(web_docs)
        print(f"\n  embedder = {embedder_name}, docs={pipeline.store.counts()['documents']}")
        for label, question, needle in CROSS_LINGUAL:
            hits = pipeline.retrieve(question, top_k=5)
            web_hits = [u for u in hits.units if u.chunk.metadata.get("origin") == "web"]
            matched = [u for u in hits.units if needle in u.chunk.text.lower()]
            print(f"    {label}: {question[:34]:<34} web_in_top5={len(web_hits)} matched={len(matched)}")
        pipeline.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())