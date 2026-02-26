#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

DEFAULT_WEIGHTS = {
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


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


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
    if "```" in t or re.search(r"\b(def|class|function|const|let|var|import|return)\b", t):
        return "code"
    if re.search(r"(^#\s)|(^-\s)|(^\d+\.\s)", t, re.M):
        return "markdown"
    return "text"


def _weights_or_default(weights: Dict[str, float] | None) -> Dict[str, float]:
    active = dict(DEFAULT_WEIGHTS)
    if weights:
        active.update(weights)
    total = sum(active.values()) or 1.0
    # normalize to 1.0
    return {k: v / total for k, v in active.items()}


def _score_dimensions(submission: str, rubric_keywords: List[str]) -> Tuple[str, Dict[str, float]]:
    fmt = _detect_format(submission)
    txt = submission.strip()
    tokens = _tokenize(txt)

    completeness = _clamp(len(tokens) / 120.0)
    if fmt in ("json", "markdown", "code"):
        completeness = _clamp(completeness + 0.12)

    if fmt == "json":
        try:
            parsed = json.loads(txt)
            format_compliance = 0.95 if isinstance(parsed, (dict, list)) else 0.8
        except Exception:
            format_compliance = 0.2
    elif fmt == "markdown":
        format_compliance = 0.82 if re.search(r"(^#\s)|(^-\s)|(^\d+\.\s)", txt, re.M) else 0.55
    elif fmt == "code":
        format_compliance = 0.82 if re.search(r"[{}();]|\n\s{2,}\w", txt) else 0.58
    else:
        format_compliance = 0.72

    hits = sum(1 for k in rubric_keywords if k.lower() in txt.lower())
    coverage = _clamp(hits / max(1, len(rubric_keywords)))

    punct = len(re.findall(r"[.!?]", txt))
    symbol_noise = len(re.findall(r"[^\w\s\.,;:!?\-\(\)\[\]\{\}'\"/\\]", txt))
    clarity = 0.62 + min(0.22, punct / 20.0) - min(0.30, symbol_noise / max(1, len(txt)) * 5)
    clarity = _clamp(clarity)

    validity = 0.65
    if fmt == "json":
        try:
            parsed = json.loads(txt)
            validity = 0.9 if isinstance(parsed, (dict, list)) else 0.75
        except Exception:
            validity = 0.12
    elif fmt == "code":
        brackets_ok = txt.count("(") == txt.count(")") and txt.count("{") == txt.count("}")
        validity = 0.8 if brackets_ok else 0.45

    return fmt, {
        "completeness": round(completeness, 4),
        "format_compliance": round(format_compliance, 4),
        "coverage": round(coverage, 4),
        "clarity": round(clarity, 4),
        "validity": round(validity, 4),
    }


def _rating(weighted: float) -> str:
    if weighted >= 0.85:
        return "excellent"
    if weighted >= 0.7:
        return "good"
    if weighted >= 0.5:
        return "fair"
    return "poor"


def _feedback(scores: Dict[str, float]) -> List[str]:
    fb = []
    messages = {
        "completeness": "内容不够完整，建议补充上下文、步骤和结果。",
        "format_compliance": "格式规范不足，建议按目标格式（JSON/Markdown/Code）整理。",
        "coverage": "覆盖面偏低，建议补全 rubric 关键词相关内容。",
        "clarity": "表达清晰度一般，建议拆句并减少噪声字符。",
        "validity": "有效性偏弱，建议修复结构错误或语法问题。",
    }
    for k, v in scores.items():
        if v < 0.5:
            fb.append(messages[k])
    if not fb:
        fb.append("整体质量较好，已覆盖主要评分维度。")
    return fb


def score_submission(
    submission: str,
    rubric_keywords: List[str] | None = None,
    threshold: float = 0.7,
    weights: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    rubric_keywords = rubric_keywords or ["summary", "steps", "result"]
    active_weights = _weights_or_default(weights)

    _, scores = _score_dimensions(submission, rubric_keywords)
    weighted = sum(scores[k] * active_weights[k] for k in active_weights)

    return ScoreResult(
        weighted_score=weighted,
        quality_rating=_rating(weighted),
        scores=scores,
        feedback=_feedback(scores),
        pass_threshold=weighted >= threshold,
    ).to_dict()


def evaluate_against_ground_truth(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """dataset item: {submission:str, expected_score:float, rubric_keywords?:list[str]}"""
    abs_errors: List[float] = []
    results = []
    for item in dataset:
        out = score_submission(item["submission"], item.get("rubric_keywords"))
        err = abs(out["weighted_score"] - float(item["expected_score"]))
        abs_errors.append(err)
        results.append(
            {
                "expected": round(float(item["expected_score"]), 4),
                "predicted": out["weighted_score"],
                "abs_error": round(err, 4),
                "quality_rating": out["quality_rating"],
            }
        )

    mae = sum(abs_errors) / max(1, len(abs_errors))
    max_error = max(abs_errors) if abs_errors else 0.0
    within_point_05 = sum(1 for e in abs_errors if e <= 0.05) / max(1, len(abs_errors))

    return {
        "count": len(dataset),
        "mae": round(mae, 4),
        "max_error": round(max_error, 4),
        "within_point_05_ratio": round(within_point_05, 4),
        "results": results,
    }


def benchmark_100(submissions: List[str]) -> float:
    start = time.time()
    for s in submissions[:100]:
        score_submission(s)
    return time.time() - start
