"""Command line interface: doctor / index / query / eval / demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag.eval import evaluate_generation, evaluate_retrieval, load_dataset, report
from rag.pipeline import build_pipeline, load_documents


def _print_answer(answer, show_trace: bool) -> None:
    print(answer.text or "(no answer produced)")
    if answer.citations:
        print("\nSources")
        for citation in answer.citations:
            print(f"[{citation.index}] {citation.doc_id} :: {citation.heading}")
    if answer.groundedness is not None:
        print(f"\ngroundedness={answer.groundedness:.3f} answerable={answer.answerable}")
    if answer.evidence:
        ev = answer.evidence
        print(
            f"evidence: type={ev['answer_type']} sufficient={ev['answerable']} "
            f"support={ev['support']} entities_missing={ev['missing_entities']}"
        )
    if answer.delivery:
        print("delivery: " + " ".join(f"{k}={v}" for k, v in answer.delivery.items()))
    usage = getattr(answer.trace, "usage", None)
    if usage:
        print("usage: " + " ".join(f"{k}={v}" for k, v in usage.items() if v))
    if show_trace:
        print("\ntrace: " + answer.trace.summary())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag", description="Offline-first RAG framework")
    parser.add_argument("--config", default=None, help="YAML config path")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="show detected embedder/LLM/web capability")

    p_index = sub.add_parser("index", help="index files or directories")
    p_index.add_argument("paths", nargs="+")
    p_index.add_argument("--strategy", default=None, choices=["structural", "semantic", "hierarchical"])

    p_query = sub.add_parser("query", help="ask a question")
    p_query.add_argument("question")
    p_query.add_argument("--k", type=int, default=None)
    p_query.add_argument("--no-agentic", action="store_true")
    p_query.add_argument("--trace", action="store_true")
    p_query.add_argument("--doc", action="append", default=None, help="restrict to a doc_id (repeatable)")

    p_search = sub.add_parser("search", help="retrieve only, no generation")
    p_search.add_argument("question")
    p_search.add_argument("--k", type=int, default=8)
    p_search.add_argument("--json", action="store_true")

    p_eval = sub.add_parser("eval", help="run a JSONL eval dataset")
    p_eval.add_argument("dataset")
    p_eval.add_argument("--k", type=int, default=8)
    p_eval.add_argument("--generation", action="store_true", help="also score faithfulness/relevancy")

    p_demo = sub.add_parser("demo", help="index examples/corpus and answer sample questions")
    p_demo.add_argument("--corpus", default="examples/corpus")

    p_stress = sub.add_parser("stress", help="measure retrieval under a deliberately degraded corpus")
    p_stress.add_argument("dataset")
    p_stress.add_argument("--paths", nargs="*", default=None)
    p_stress.add_argument("--k", type=int, default=5)
    p_stress.add_argument("--levels", default="0,1,2,3", help="comma separated corruption levels")
    p_stress.add_argument("--operators", default="typo,truncate,duplicate,shuffle,noise,conflict")
    p_stress.add_argument("--seed", type=int, default=20261006)

    p_web = sub.add_parser("webtest", help="probe live web retrieval")
    p_web.add_argument("query", nargs="?", default="retrieval augmented generation latest techniques")
    p_web.add_argument("--backend", default="auto", choices=["auto", "bing", "duckduckgo", "arxiv", "tavily", "brave"])
    p_web.add_argument("--k", type=int, default=5)

    args = parser.parse_args(argv)
    pipeline = build_pipeline(args.config)

    if args.command == "doctor":
        print(json.dumps(pipeline.doctor(), indent=2, ensure_ascii=False))
        return 0

    if args.command == "webtest":
        web = pipeline.web
        web.backend = args.backend
        print("network available:", web.available(), "| backend:", args.backend)
        if not web.available():
            print("web retrieval disabled: sandbox has no network access")
            return 1
        for hit in web.search(args.query, max_results=args.k):
            print(f"- {hit.title}\n  {hit.url}\n  {hit.snippet[:160]}")
        return 0

    if args.command == "demo":
        corpus = Path(args.corpus)
        if not corpus.exists():
            print(f"corpus not found: {corpus}", file=sys.stderr)
            return 1
        stats = pipeline.index_paths([corpus])
        print(f"indexed docs={stats.documents} chunks={stats.chunks} embedder={stats.embedder} in {stats.elapsed_s}s")
        for question in (
            "hybrid retrieval 为什么比单一 BM25 更好",
            "what is late interaction reranking",
            "如何评估检索系统的质量",
        ):
            print("\nQ:", question)
            _print_answer(pipeline.answer(question), show_trace=True)
        return 0

    stats = pipeline.index_paths(args.paths) if args.command == "index" else _seed_default_corpus(pipeline)
    if args.command == "index":
        if args.strategy:
            pipeline.config.chunking.strategy = args.strategy
            stats = pipeline.index_documents(load_documents(args.paths))
        print(
            f"documents={stats.documents} chunks={stats.chunks} vectors={stats.embedded} "
            f"skipped_unchanged={stats.skipped} embedder={stats.embedder} time={stats.elapsed_s}s"
        )
        return 0

    if args.command == "search":
        retrieved = pipeline.retrieve(args.question, top_k=args.k)
        if args.json:
            print(json.dumps([{"chunk_id": u.chunk.chunk_id, "score": round(u.final_score, 5), "text": u.chunk.text[:200]} for u in retrieved.units], indent=2, ensure_ascii=False))
            return 0
        for unit in retrieved.units:
            print(f"{unit.rank}. [{unit.chunk.doc_id}] {unit.final_score:.4f}  {unit.chunk.text[:120].replace(chr(10), ' ')}")
        print("\ntrace: " + retrieved.trace.summary())
        return 0

    if args.command == "stress":
        return _run_stress(args, pipeline)

    if args.command == "eval":
        _seed_default_corpus(pipeline)
        cases = load_dataset(args.dataset)
        retrieval_metrics = evaluate_retrieval(cases, lambda q: pipeline.retrieve(q).chunk_ids, k=args.k)
        print("retrieval")
        print(report(retrieval_metrics))
        if args.generation:
            print("\ngeneration")
            print(report(evaluate_generation(cases, pipeline.answer, pipeline.embedder)))
        return 0

    filters = {"doc_ids": args.doc} if getattr(args, "doc", None) else None
    _print_answer(pipeline.answer(args.question, agentic=not args.no_agentic, filters=filters), show_trace=args.trace)
    return 0


def _run_stress(args, pipeline) -> int:
    """Re-index the corpus once per corruption level into throwaway stores."""

    import tempfile

    from rag.config import RAGConfig
    from rag.pipeline import RAGPipeline
    from rag.stress import StressConfig, degradation_table, stress_curve

    cases = load_dataset(args.dataset)
    paths = args.paths or _default_corpus()
    if not paths:
        print("no corpus found; pass --paths", file=sys.stderr)
        return 2
    documents = load_documents(paths)
    config = pipeline.config
    cfg = StressConfig(
        operators=tuple(o.strip() for o in args.operators.split(",") if o.strip()),
        levels=tuple(int(x) for x in args.levels.split(",") if x.strip()),
        seed=args.seed,
    )
    scratch = Path(tempfile.mkdtemp(prefix="ragstress-"))
    counter = {"n": 0}

    def evaluate(corpus):
        level_config = RAGConfig.from_dict(config.to_dict())
        level_config.db_path = str(scratch / f"level-{counter['n']}.db")
        level_config.embed_cache_path = str(scratch / "embeddings.db")
        counter["n"] += 1
        probe = RAGPipeline(level_config)
        try:
            probe.index_documents(corpus)
            return evaluate_retrieval(cases, lambda q: probe.retrieve(q).chunk_ids, k=args.k)
        finally:
            probe.close()

    rows = stress_curve(documents, cases, evaluate, cfg)
    for row in rows:
        print(f"level {row['level']}  docs={row['documents']}  " + "  ".join(
            f"{k}={v}" for k, v in row.items() if k not in {"level", "documents"}
        ))
    metric = f"recall@{args.k}"
    print()
    print(degradation_table(rows, metric))
    return 0


def _seed_default_corpus(pipeline) -> object:
    """Only fall back to the bundled corpus for an empty store.

    Indexing it unconditionally would inject examples/corpus into a knowledge base
    the user deliberately built somewhere else, and the answers would cite it.
    """

    if pipeline.store.counts()["documents"] == 0:
        print("store is empty: indexing the bundled corpus (examples/corpus)", file=sys.stderr)
        return pipeline.index_paths(_default_corpus())
    return pipeline.index_documents([])


def _default_corpus() -> list[str]:
    candidate = Path("examples/corpus")
    return [candidate] if candidate.exists() else []


if __name__ == "__main__":
    raise SystemExit(main())