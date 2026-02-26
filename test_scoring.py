import json
import unittest

from scoring import QualityScorer


class TestQualityScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = QualityScorer(pass_threshold=0.7)

    def test_json_submission(self):
        submission = json.dumps(
            {
                "goal": "Build scoring",
                "approach": "Weighted rubric",
                "result": "0-1 output",
                "risk": "edge formats",
                "timeline": "2 days",
                "test": "20-case benchmark",
            }
        )
        out = self.scorer.score(submission)
        self.assertIn("weighted_score", out)
        self.assertIn("scores", out)
        self.assertGreaterEqual(out["scores"]["format_compliance"], 0.95)

    def test_markdown_submission(self):
        submission = """# Proposal\n- goal: build engine\n- approach: rubric scoring\n- result: reliable scores\n- risk: malformed input\n- timeline: 2d\n- test: benchmark suite\n"""
        out = self.scorer.score(submission)
        self.assertEqual(out["quality_rating"] in {"good", "excellent", "fair", "poor"}, True)
        self.assertTrue(isinstance(out["feedback"], list))

    def test_benchmark_100_under_10s(self):
        samples = [
            f"goal approach result risk timeline test sample {i}" for i in range(100)
        ]
        b = self.scorer.benchmark(samples)
        self.assertEqual(b["count"], 100)
        self.assertLess(b["elapsed_seconds"], 10.0)


if __name__ == "__main__":
    unittest.main()
