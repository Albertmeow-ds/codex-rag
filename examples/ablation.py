"""Ablation harness: isolate what each channel, reranker and chunker contributes."""

import sys

sys.path.insert(0, ".")

from pathlib import Path

from rag.config import RAGConfig
from rag.eval import evaluate_retrieval, load_dataset, report
from rag.pipeline import RAGPipeline

cases = load_dataset("examples/eval.jsonl")
SECTIONS = {"chunking", "retrieval", "generation", "agentic"}


def build(label: str, **overrides) -> RAGPipeline:
    config = RAGConfig.from_file("config.yaml")
    config.llm = "none"
    for key, value in overrides.items():
        if key in SECTIONS:
            target = getattr(config, key)
            for sub_key, sub_value in value.items():
                if isinstance(getattr(target, sub_key), tuple) and isinstance(sub_value, list):
                    sub_value = tuple(sub_value)
                setattr(target, sub_key, sub_value)
        else:
            setattr(config, key, value)
    slug = label.replace(" ", "_").replace("(", "").replace(")", "")
    db = Path("data") / f"ablation_{slug}.db"
    if db.exists():
        db.unlink()
    config.db_path = str(db)
    pipeline = RAGPipeline(config)
    pipeline.index_paths(["examples/corpus"])
    return pipeline


def run(label: str, **overrides) -> RAGPipeline:
    pipeline = build(label, **overrides)
    metrics = evaluate_retrieval(cases, lambda q: pipeline.retrieve(q).chunk_ids, k=5)
    print(f"{label:<36} {report(metrics).replace(chr(10), ' | ')}")
    return pipeline


run("bm25 only", retrieval={"channels": ["bm25"]}, reranker="none")
run("dense only (hashed)", retrieval={"channels": ["dense"]}, reranker="none")
run("hybrid rrf (hashed)", reranker="none")
run("hybrid + late rerank", reranker="late")
run("hybrid + heuristic rerank", reranker="heuristic")
run("hybrid + late channel + rerank", retrieval={"channels": ["bm25", "dense", "late"]}, reranker="late")
run("weighted fusion", retrieval={"fusion": "weighted"}, reranker="late")
run("structural chunking", chunking={"strategy": "structural"}, reranker="late")
run("semantic chunking", chunking={"strategy": "semantic"}, reranker="late")
run("sentence window radius=2", retrieval={"sentence_window_radius": 2, "expand_parents": False}, reranker="late")
run("lsa embedder", embedder="lsa", reranker="late")
run("lsa + heuristic rerank", embedder="lsa", reranker="heuristic")

print()
pipeline = build("diagnostic", reranker="none")
query = "RRF 融合相比加权分数融合有什么优缺点"
before = [(u.chunk.chunk_id, round(u.score, 5)) for u in pipeline.retrieve(query).units]
pipeline.config.reranker = "late"
pipeline.config.retrieval.top_k = 8
after = [(u.chunk.chunk_id, round(u.final_score, 4)) for u in pipeline.retrieve(query).units]
print("first-stage order:", [c.split('/')[-1] for c, _ in before])
print("after late rerank:", [c.split('/')[-1] for c, _ in after])
pipeline.close()