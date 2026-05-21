"""Trusted DSL interpreter for deterministic fitness functions."""

from __future__ import annotations

import math
from typing import Any


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if not math.isfinite(value):
        return low
    return max(low, min(high, value))


def flatten_metrics(evidence) -> dict[str, Any]:
    metrics = dict(evidence.raw_metrics)
    validation = evidence.validation_summary
    for key in (
        "num_cases",
        "passed",
        "failed",
        "pass_rate",
        "hidden_num_cases",
        "hidden_passed",
        "hidden_failed",
        "hidden_pass_rate",
    ):
        value = getattr(validation, key, None)
        if value is not None:
            metrics.setdefault(key, value)
    metrics.setdefault("combined_score_before_rerank", evidence.combined_score_before_rerank)
    return metrics


class FitnessDSLInterpreter:
    def __init__(self, config=None):
        self.config = config

    def score(self, spec, evidence) -> float:
        metrics = flatten_metrics(evidence)
        gate = self._compute_gate(spec.formula_dsl.get("gate"), metrics)
        total = 0.0
        for component in spec.formula_dsl.get("components", []):
            total += float(component.get("weight", 1.0)) * self._transform(
                self._metric_value(metrics, component), component
            )
        for penalty in spec.formula_dsl.get("penalties", []):
            total -= float(penalty.get("weight", 1.0)) * self._transform(
                self._metric_value(metrics, penalty), penalty
            )
        return _clamp(gate * total)

    def _metric_value(self, metrics: dict[str, Any], item: dict[str, Any]) -> float:
        metric = item["metric"]
        if metric in metrics and isinstance(metrics[metric], (int, float)):
            return float(metrics[metric])
        if "default" in item:
            return float(item["default"])
        raise ValueError(f"Metric {metric!r} not available for fitness DSL")

    def _transform(self, value: float, item: dict[str, Any]) -> float:
        transform = item.get("transform", "identity")
        if transform == "identity":
            return _clamp(value)
        if transform == "saturating":
            return _clamp(1.0 - math.exp(-max(value, 0.0)))
        if transform == "log1p":
            return _clamp(math.log1p(max(value, 0.0)) / math.log(2.0))
        if transform == "inverse":
            return _clamp(1.0 / (1.0 + max(value, 0.0)))
        if transform == "capped":
            return _clamp(min(max(value, 0.0), float(item.get("cap", 1.0))))
        if transform == "boolean":
            return 1.0 if bool(value) else 0.0
        raise ValueError(f"Unsupported fitness DSL transform: {transform}")

    def _compute_gate(self, gate: dict[str, Any] | None, metrics: dict[str, Any]) -> float:
        if not gate or gate.get("kind") in {None, "none"}:
            return 1.0
        metric = gate.get("metric")
        value = float(metrics.get(metric, gate.get("default", 0.0)) or 0.0)
        kind = gate.get("kind")
        threshold = float(gate.get("threshold", 1.0) or 1.0)
        if kind == "hard_threshold":
            return 0.0 if value < threshold else 1.0
        if kind == "soft_threshold":
            if value >= threshold:
                return 1.0
            below_scale = float(gate.get("below_scale", 0.1))
            return _clamp(below_scale * value / threshold)
        if kind == "linear":
            return _clamp(value)
        raise ValueError(f"Unsupported fitness DSL gate: {kind}")
