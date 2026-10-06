"""Budget accounting: the loop must be able to say what it spent, and stop when spent out."""

import unittest

from rag.agentic import AgenticConfig, AgenticRAG, RetrievalGrader
from rag.generate import ExtractiveAnswerer
from rag.types import Chunk, RetrievedUnit
from rag.usage import BudgetConfig, CountingLLM, Usage


class FakeLLM:
    name = "fake"

    def __init__(self, reply: str = "ok", last_usage: dict | None = None) -> None:
        self.reply = reply
        self.calls = 0
        self.last_usage = last_usage or {}

    def complete(self, prompt: str, system: str = "", temperature: float = 0.2, max_tokens: int = 1200) -> str:
        self.calls += 1
        return self.reply


class FakeRetriever:
    def __init__(self, units):
        self._units = units
        from rag.types import Trace
        self.trace = Trace()

    def retrieve(self, query, top_k=None, filters=None):
        from rag.types import Retrieved
        return Retrieved(query=query, units=list(self._units))


class UsageTest(unittest.TestCase):
    def test_counting_llm_estimates_when_server_is_silent(self):
        usage = Usage()
        counted = CountingLLM(FakeLLM("a short reply here"), usage)
        counted.complete("a prompt of some length", system="sys")
        self.assertEqual(usage.llm_calls, 1)
        self.assertGreater(usage.prompt_tokens, 0)
        self.assertGreater(usage.completion_tokens, 0)

    def test_counting_llm_prefers_reported_usage(self):
        usage = Usage()
        counted = CountingLLM(FakeLLM("x", {"prompt_tokens": 123, "completion_tokens": 45}), usage)
        counted.complete("hello")
        self.assertEqual(usage.prompt_tokens, 123)
        self.assertEqual(usage.completion_tokens, 45)

    def test_budget_reports_the_axis_that_blew_up(self):
        usage = Usage(llm_calls=3)
        self.assertEqual(BudgetConfig(max_llm_calls=3).exceeded(usage), "llm_calls>=3")
        self.assertEqual(BudgetConfig(max_llm_calls=9).exceeded(usage), "")

    def test_merge_accumulates(self):
        a, b = Usage(llm_calls=1, rounds=2), Usage(llm_calls=3, rounds=5)
        a.merge(b)
        self.assertEqual((a.llm_calls, a.rounds), (4, 7))

    def test_loop_stops_at_the_llm_call_budget(self):
        unit = RetrievedUnit(chunk=Chunk("c1", "doc", "检索权重可以按嵌入模型自动切换。"), score=1.0)
        llm = FakeLLM('{"label": "irrelevant", "score": 0.0}')
        usage = Usage()
        agent = AgenticRAG(
            FakeRetriever([unit]),
            ExtractiveAnswerer(),
            config=AgenticConfig(max_rounds=5, weak_coverage_threshold=0.99, max_llm_calls=1),
            grader=RetrievalGrader(CountingLLM(llm, usage)),
            usage=usage,
        )
        answer = agent.run("完全无关的问题 xyz")
        self.assertLessEqual(usage.llm_calls, 1)
        self.assertTrue(any(stage["stage"] == "budget" for stage in answer.trace.stages))


class LexicalStopTest(unittest.TestCase):
    def test_no_lexical_hit_stops_the_loop_without_refusing_when_web_is_available(self):
        from rag.agentic import AgenticConfig, AgenticRAG, RetrievalGrader
        from rag.generate import ExtractiveAnswerer

        dense_only = RetrievedUnit(chunk=Chunk("c1", "doc", "语义相近但没有任何词面重合的段落"), score=1.5)
        dense_only.channels = {"dense": 1.5}

        def run(stop: bool):
            usage = Usage()
            agent = AgenticRAG(
                FakeRetriever([dense_only]),
                ExtractiveAnswerer(),
                config=AgenticConfig(max_rounds=3, weak_coverage_threshold=0.99, stop_when_no_lexical_hit=stop),
                usage=usage,
            )
            return agent.run("完全换个说法的问题"), usage

        _off, usage_off = run(False)
        stopped, usage_on = run(True)
        self.assertGreater(usage_off.rounds, 1)
        self.assertEqual(usage_on.rounds, 1)
        self.assertTrue(any(stage["stage"] == "stop" and stage.get("reason") == "no_lexical_hit" for stage in stopped.trace.stages))

    def test_a_lexical_hit_disables_the_stop(self):
        from rag.agentic import AgenticConfig, AgenticRAG
        from rag.generate import ExtractiveAnswerer

        unit = RetrievedUnit(chunk=Chunk("c1", "doc", "包含查询词的段落"), score=2.0)
        unit.channels = {"bm25": 0.9, "dense": 1.5}
        usage = Usage()
        agent = AgenticRAG(
            FakeRetriever([unit]),
            ExtractiveAnswerer(),
            config=AgenticConfig(max_rounds=3, weak_coverage_threshold=0.99, stop_when_no_lexical_hit=True),
            usage=usage,
        )
        answer = agent.run("包含查询词")
        self.assertFalse(
            any(stage.get("reason") == "no_lexical_hit" for stage in answer.trace.stages),
            "a candidate with a BM25 contribution must not trip the lexical stop",
        )
        self.assertTrue(answer.units)


if __name__ == "__main__":
    unittest.main()