#!/usr/bin/env python3
import json
import unittest

from quality_scorer import score_submission, benchmark_100


class TestQualityScorer(unittest.TestCase):
    def test_output_schema(self):
        out = score_submission('{"summary":"ok","steps":[1,2],"result":"done"}')
        self.assertIn("weighted_score", out)
        self.assertIn("quality_rating", out)
        self.assertIn("scores", out)
        self.assertIn("feedback", out)
        self.assertIn("pass_threshold", out)
        self.assertEqual(set(out["scores"].keys()), {"completeness", "format_compliance", "coverage", "clarity", "validity"})

    def test_formats(self):
        samples = [
            '{"summary":"hello","result":"x"}',
            '# Title\n- step 1\n- step 2\nresult done',
            'def run(x):\n    return x+1',
            'plain text summary steps and result',
        ]
        for s in samples:
            out = score_submission(s)
            self.assertGreaterEqual(out["weighted_score"], 0.0)
            self.assertLessEqual(out["weighted_score"], 1.0)

    def test_benchmark(self):
        subs = [json.dumps({"summary": "a", "steps": [1,2,3], "result": "ok"}) for _ in range(100)]
        sec = benchmark_100(subs)
        self.assertLess(sec, 10.0)


if __name__ == "__main__":
    unittest.main()
