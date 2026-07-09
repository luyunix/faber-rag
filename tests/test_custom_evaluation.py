import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.libs.evaluator.custom_evaluator import CustomEvaluator
from src.observability.evaluation.eval_runner import EvalRunner, load_test_set


class CustomEvaluatorTests(unittest.TestCase):
    def test_scores_chunk_id_by_first_relevant_rank(self):
        evaluator = CustomEvaluator(metrics=["hit_rate", "mrr"])

        metrics = evaluator.evaluate(
            query="Azure 模型怎么配置？",
            retrieved_chunks=[
                {"chunk_id": "wrong", "text": "价格说明"},
                {"chunk_id": "correct", "text": "配置说明"},
            ],
            ground_truth={"ids": ["correct"]},
        )

        self.assertEqual(metrics, {"hit_rate": 1.0, "mrr": 0.5})

    def test_matches_expected_source_by_filename(self):
        evaluator = CustomEvaluator()

        metrics = evaluator.evaluate(
            query="Azure 模型怎么配置？",
            retrieved_chunks=[
                SimpleNamespace(
                    chunk_id="chunk-a",
                    text="无关内容",
                    metadata={"source_path": "/data/docs/其他资料.pdf"},
                ),
                SimpleNamespace(
                    chunk_id="chunk-b",
                    text="配置说明",
                    metadata={"source_path": "/data/docs/Azure部署指南.pdf"},
                ),
            ],
            ground_truth={"sources": ["Azure部署指南.pdf"]},
        )

        self.assertEqual(metrics, {"hit_rate": 1.0, "mrr": 0.5})

    def test_returns_zero_for_empty_retrieval(self):
        evaluator = CustomEvaluator()

        metrics = evaluator.evaluate(
            query="不存在的内容",
            retrieved_chunks=[],
            ground_truth={"sources": ["资料.pdf"]},
        )

        self.assertEqual(metrics, {"hit_rate": 0.0, "mrr": 0.0})

    def test_rejects_missing_ground_truth(self):
        evaluator = CustomEvaluator()

        with self.assertRaisesRegex(ValueError, "expected_chunk_ids 或 expected_sources"):
            evaluator.evaluate(
                query="Azure 模型怎么配置？",
                retrieved_chunks=[{"chunk_id": "chunk-a"}],
                ground_truth=None,
            )


class EvalRunnerTests(unittest.TestCase):
    def _write_test_set(self, data):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "golden.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return path

    def test_load_test_set_accepts_single_source_string(self):
        path = self._write_test_set({
            "test_cases": [{
                "query": "Azure 模型怎么配置？",
                "expected_sources": "Azure部署指南.pdf",
            }]
        })

        test_cases = load_test_set(path)

        self.assertEqual(test_cases[0].expected_sources, ["Azure部署指南.pdf"])

    def test_rejects_unlabelled_custom_cases(self):
        path = self._write_test_set({
            "test_cases": [{"query": "没有标签的问题"}]
        })
        runner = EvalRunner(evaluator=CustomEvaluator(), hybrid_search=object())

        with self.assertRaisesRegex(ValueError, "缺少标签的题号：1"):
            runner.run(path)

    def test_uses_expected_sources_for_aggregate_metrics(self):
        path = self._write_test_set({
            "test_cases": [{
                "query": "Azure 模型怎么配置？",
                "expected_sources": ["Azure部署指南.pdf"],
            }]
        })

        class Search:
            def search(self, query, top_k):
                return [
                    {"chunk_id": "wrong", "text": "价格", "metadata": {"source": "价格表.pdf"}},
                    {
                        "chunk_id": "correct",
                        "text": "配置 provider 和 model",
                        "metadata": {"source_path": "docs/Azure部署指南.pdf"},
                    },
                ]

        report = EvalRunner(
            evaluator=CustomEvaluator(),
            hybrid_search=Search(),
        ).run(path, top_k=10)

        self.assertEqual(report.aggregate_metrics, {"hit_rate": 1.0, "mrr": 0.5})
        self.assertEqual(report.query_results[0].retrieved_chunk_ids, ["wrong", "correct"])


if __name__ == "__main__":
    unittest.main()
