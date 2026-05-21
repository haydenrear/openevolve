"""Score normalization helpers for agentic fitness epochs."""

from __future__ import annotations

import math


def rank_percentile(
    scores: dict[str, float], floor: float = 0.0, ceiling: float = 1.0
) -> dict[str, float]:
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if not ordered:
        return {}
    if len(ordered) == 1:
        return {ordered[0][0]: ceiling}
    return {
        program_id: floor + (1.0 - idx / (len(ordered) - 1)) * (ceiling - floor)
        for idx, (program_id, _) in enumerate(ordered)
    }


def minmax(scores: dict[str, float], floor: float = 0.0, ceiling: float = 1.0) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    low = min(values)
    high = max(values)
    if high == low:
        return {program_id: ceiling for program_id in scores}
    return {
        program_id: floor + ((score - low) / (high - low)) * (ceiling - floor)
        for program_id, score in scores.items()
    }


def zscore_sigmoid(
    scores: dict[str, float], floor: float = 0.0, ceiling: float = 1.0
) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(1, len(values))
    std = math.sqrt(variance) or 1.0
    return {
        program_id: floor + (1.0 / (1.0 + math.exp(-((score - mean) / std)))) * (ceiling - floor)
        for program_id, score in scores.items()
    }


def normalize_scores(
    scores: dict[str, float], method: str, floor: float = 0.0, ceiling: float = 1.0
) -> dict[str, float]:
    if method == "rank_percentile":
        return rank_percentile(scores, floor, ceiling)
    if method == "minmax":
        return minmax(scores, floor, ceiling)
    if method == "zscore_sigmoid":
        return zscore_sigmoid(scores, floor, ceiling)
    raise ValueError(f"Unsupported normalization method: {method}")
