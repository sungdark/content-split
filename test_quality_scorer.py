#!/usr/bin/env python3
import json
import unittest

from quality_scorer import benchmark_100, evaluate_against_ground_truth, score_submission


class TestQualityScorer(unittest.TestCase):
    def test_output_schema(self):
        out = score_submission('{"summary":"ok","steps":[1,2],"result":"done"}')
        self.assertIn("weighted_score", out)
        self.assertIn("quality_rating", out)
        self.assertIn("scores", out)
        self.assertIn("feedback", out)
        self.assertIn("pass_threshold", out)
        self.assertEqual(
            set(out["scores"].keys()),
            {"completeness", "format_compliance", "coverage", "clarity", "validity"},
        )

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
        subs = [json.dumps({"summary": "a", "steps": [1, 2, 3], "result": "ok"}) for _ in range(100)]
        sec = benchmark_100(subs)
        self.assertLess(sec, 10.0)

    def test_ground_truth_alignment(self):
        # 20 samples with expected scores that this heuristic should approximate.
        dataset = [
            {"submission": '{"summary":"clear summary","steps":["a","b"],"result":"done"}', "expected_score": 0.84},
            {"submission": '{"summary":"short","result":"ok"}', "expected_score": 0.68},
            {"submission": '# Report\n- summary\n- steps\n- result', "expected_score": 0.78},
            {"submission": 'summary: yes. steps: yes. result: yes.', "expected_score": 0.72},
            {"submission": 'def run(x):\n    # summary\n    return x', "expected_score": 0.66},
            {"submission": '{"foo":1}', "expected_score": 0.53},
            {"submission": 'random words only', "expected_score": 0.38},
            {"submission": '```python\ndef x():\n    return 1\n```', "expected_score": 0.58},
            {"submission": '1. summary\n2. steps\n3. result', "expected_score": 0.76},
            {"submission": '{"summary":"ok","steps":[1,2,3],"result":"final"}', "expected_score": 0.86},
            {"submission": '{"summary":"ok","steps":[],"result":"done"}', "expected_score": 0.77},
            {"submission": '### Header\nresult only', "expected_score": 0.54},
            {"submission": 'summary and steps but no final result', "expected_score": 0.61},
            {"submission": '{"summary":"detailed","steps":["a","b","c"],"result":"good","notes":"clear"}', "expected_score": 0.88},
            {"submission": 'def bad(x:\n    return x', "expected_score": 0.44},
            {"submission": '# title\n- summary\ntext text text\nresult', "expected_score": 0.67},
            {"submission": '{"summary":"s","steps":["1"],"result":"r","extra":"..."}', "expected_score": 0.79},
            {"submission": 'plain text with summary steps result and clear ending.', "expected_score": 0.74},
            {"submission": '{"summary":"none"}', "expected_score": 0.56},
            {"submission": 'tiny', "expected_score": 0.3},
        ]
        metrics = evaluate_against_ground_truth(dataset)
        self.assertEqual(metrics["count"], 20)
        # Heuristic baseline keeps average error reasonably small; this can be tightened
        # if/when a trained calibrator is introduced.
        self.assertLessEqual(metrics["mae"], 0.15)
        self.assertGreaterEqual(metrics["within_point_05_ratio"], 0.10)


if __name__ == "__main__":
    unittest.main()
