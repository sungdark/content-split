from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

WEIGHTS: Dict[str, float] = {
    "completeness": 0.30,
    "format_compliance": 0.20,
    "coverage": 0.25,
    "clarity": 0.15,
    "validity": 0.10,
}


@dataclass
class ScoreResult:
    weighted_score: float
    quality_rating: str
    scores: Dict[str, float]
    feedback: List[str]
    pass_threshold: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "weighted_score": round(self.weighted_score, 4),
            "quality_rating": self.quality_rating,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "feedback": self.feedback,
            "pass_threshold": self.pass_threshold,
        }


class QualityScorer:
    def __init__(self, pass_threshold: float = 0.7):
        self.pass_threshold = pass_threshold

    def score(self, submission: str) -> Dict[str, Any]:
        submission = submission or ""
        fmt = self._detect_format(submission)
        sections = self._extract_sections(submission, fmt)

        scores = {
            "completeness": self._score_completeness(submission, sections),
            "format_compliance": self._score_format_compliance(submission, fmt),
            "coverage": self._score_coverage(submission, sections),
            "clarity": self._score_clarity(submission),
            "validity": self._score_validity(submission, fmt),
        }

        weighted = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)
        feedback = self._feedback(scores, fmt)

        return ScoreResult(
            weighted_score=weighted,
            quality_rating=self._rating(weighted),
            scores=scores,
            feedback=feedback,
            pass_threshold=weighted >= self.pass_threshold,
        ).as_dict()

    def benchmark(self, submissions: List[str]) -> Dict[str, Any]:
        start = time.perf_counter()
        scorecards = [self.score(s) for s in submissions]
        elapsed = time.perf_counter() - start
        return {
            "count": len(submissions),
            "elapsed_seconds": round(elapsed, 4),
            "scorecards": scorecards,
        }

    def _detect_format(self, text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return "text"
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                json.loads(stripped)
                return "json"
            except Exception:
                pass
        if "```" in text or re.search(r"\b(def|class|function|const|let)\b", text):
            return "code"
        if re.search(r"^#{1,6}\s+", text, flags=re.M) or re.search(r"^[-*]\s+", text, flags=re.M):
            return "markdown"
        return "text"

    def _extract_sections(self, text: str, fmt: str) -> List[str]:
        if fmt == "json":
            try:
                obj = json.loads(text)
                if isinstance(obj, dict):
                    return list(obj.keys())
                return ["list"]
            except Exception:
                return []
        return re.findall(r"^#{1,6}\s+(.+)$", text, flags=re.M)

    def _score_completeness(self, text: str, sections: List[str]) -> float:
        length = len(text.split())
        section_bonus = min(len(sections) / 5.0, 1.0) * 0.3
        length_score = min(length / 220.0, 1.0) * 0.7
        return min(length_score + section_bonus, 1.0)

    def _score_format_compliance(self, text: str, fmt: str) -> float:
        if fmt == "json":
            try:
                obj = json.loads(text)
                return 1.0 if isinstance(obj, (dict, list)) else 0.7
            except Exception:
                return 0.2
        if fmt == "code":
            return 1.0 if "```" in text else 0.75
        if fmt == "markdown":
            has_heading = bool(re.search(r"^#{1,6}\s+", text, flags=re.M))
            has_list = bool(re.search(r"^[-*]\s+", text, flags=re.M))
            return 0.6 + 0.2 * has_heading + 0.2 * has_list
        return 0.85

    def _score_coverage(self, text: str, sections: List[str]) -> float:
        key_terms = ["goal", "approach", "result", "risk", "timeline", "test"]
        txt = text.lower()
        hits = sum(1 for k in key_terms if k in txt)
        section_factor = min(len(sections) / 4.0, 1.0)
        return min((hits / len(key_terms)) * 0.7 + section_factor * 0.3, 1.0)

    def _score_clarity(self, text: str) -> float:
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0.0
        avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
        # best around 12-24 words/sentence
        if avg_len < 6:
            base = 0.5
        elif avg_len <= 24:
            base = 1.0
        elif avg_len <= 36:
            base = 0.75
        else:
            base = 0.5
        punctuation = 1.0 if re.search(r"[,;:]", text) else 0.85
        return min(base * punctuation, 1.0)

    def _score_validity(self, text: str, fmt: str) -> float:
        if fmt == "json":
            try:
                json.loads(text)
                return 1.0
            except Exception:
                return 0.1
        if fmt == "code":
            opens = text.count("{") + text.count("(") + text.count("[")
            closes = text.count("}") + text.count(")") + text.count("]")
            return 1.0 if opens == closes else 0.6
        # basic sanity check for plain text/markdown
        bad_patterns = ["lorem ipsum", "TODO", "TBD"]
        penalty = sum(1 for p in bad_patterns if p.lower() in text.lower()) * 0.1
        return max(0.9 - penalty, 0.4)

    def _rating(self, weighted: float) -> str:
        if weighted >= 0.9:
            return "excellent"
        if weighted >= 0.75:
            return "good"
        if weighted >= 0.6:
            return "fair"
        return "poor"

    def _feedback(self, scores: Dict[str, float], fmt: str) -> List[str]:
        fb = [f"Detected format: {fmt}."]
        for dim, val in scores.items():
            if val < 0.6:
                fb.append(f"Improve {dim.replace('_', ' ')} (current {val:.2f}).")
        if len(fb) == 1:
            fb.append("Strong submission across all dimensions.")
        return fb
