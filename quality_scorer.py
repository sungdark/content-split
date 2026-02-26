#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List

WEIGHTS = {
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weighted_score": round(self.weighted_score, 4),
            "quality_rating": self.quality_rating,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "feedback": self.feedback,
            "pass_threshold": self.pass_threshold,
        }


def _detect_format(text: str) -> str:
    t = text.strip()
    if not t:
        return "text"
    if (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]")):
        try:
            json.loads(t)
            return "json"
        except Exception:
            pass
    if "```" in t or re.search(r"\b(def|class|function|const|let|var|import)\b", t):
        return "code"
    if re.search(r"(^#\s)|(^-\s)|(^\d+\.\s)", t, re.M):
        return "markdown"
    return "text"


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def score_submission(submission: str, rubric_keywords: List[str] | None = None, threshold: float = 0.7) -> Dict[str, Any]:
    rubric_keywords = rubric_keywords or ["summary", "steps", "result"]
    fmt = _detect_format(submission)
    txt = submission.strip()
    tokens = _tokenize(txt)

    # Completeness: length and structure markers
    completeness = _clamp((len(tokens) / 120.0))
    if fmt in ("json", "markdown", "code"):
        completeness = _clamp(completeness + 0.15)

    # Format compliance: based on successful parse/structure patterns
    if fmt == "json":
        try:
            json.loads(txt)
            format_compliance = 0.95
        except Exception:
            format_compliance = 0.2
    elif fmt == "markdown":
        format_compliance = 0.8 if re.search(r"(^#\s)|(^-\s)|(^\d+\.\s)", txt, re.M) else 0.5
    elif fmt == "code":
        format_compliance = 0.8 if re.search(r"[{}();]|\n\s{2,}\w", txt) else 0.55
    else:
        format_compliance = 0.7

    # Coverage: keyword hit ratio
    hits = sum(1 for k in rubric_keywords if k.lower() in txt.lower())
    coverage = _clamp(hits / max(1, len(rubric_keywords)))

    # Clarity: sentence/word balance and low symbol noise
    punct = len(re.findall(r"[.!?]", txt))
    symbol_noise = len(re.findall(r"[^\w\s\.,;:!?\-\(\)\[\]\{\}'\"/\\]", txt))
    clarity = 0.6 + min(0.25, punct / 20.0) - min(0.35, symbol_noise / max(1, len(txt)) * 5)
    clarity = _clamp(clarity)

    # Validity: basic consistency and parseability clues
    validity = 0.65
    if fmt == "json":
        try:
            parsed = json.loads(txt)
            validity = 0.9 if isinstance(parsed, (dict, list)) else 0.75
        except Exception:
            validity = 0.1
    elif fmt == "code":
        brackets_ok = txt.count("(") == txt.count(")") and txt.count("{") == txt.count("}")
        validity = 0.8 if brackets_ok else 0.45

    scores = {
        "completeness": round(completeness, 4),
        "format_compliance": round(format_compliance, 4),
        "coverage": round(coverage, 4),
        "clarity": round(clarity, 4),
        "validity": round(validity, 4),
    }

    weighted = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)

    if weighted >= 0.85:
        rating = "excellent"
    elif weighted >= 0.7:
        rating = "good"
    elif weighted >= 0.5:
        rating = "fair"
    else:
        rating = "poor"

    fb = []
    for k, v in scores.items():
        if v < 0.5:
            fb.append(f"Improve {k.replace('_', ' ')}.")
    if not fb:
        fb.append("Strong overall quality across rubric dimensions.")

    return ScoreResult(
        weighted_score=weighted,
        quality_rating=rating,
        scores=scores,
        feedback=fb,
        pass_threshold=weighted >= threshold,
    ).to_dict()


def benchmark_100(submissions: List[str]) -> float:
    import time
    start = time.time()
    for s in submissions[:100]:
        score_submission(s)
    return time.time() - start
