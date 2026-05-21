"""Program evidence capture for agentic fitness reranking."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from openevolve.utils.metrics_utils import get_fitness_score


@dataclass
class ValidationCaseSummary:
    num_cases: int = 0
    passed: int = 0
    failed: int = 0
    pass_rate: Optional[float] = None
    hidden_num_cases: Optional[int] = None
    hidden_passed: Optional[int] = None
    hidden_failed: Optional[int] = None
    hidden_pass_rate: Optional[float] = None
    notable_failures: list[str] = field(default_factory=list)


@dataclass
class ProgramEvidence:
    program_id: str
    island: Optional[int]
    cell_key: Optional[str]
    code_hash: Optional[str]
    parent_program_id: Optional[str]
    generation: Optional[int]
    created_iteration: Optional[int]
    raw_metrics: dict[str, Any]
    combined_score_before_rerank: Optional[float]
    validation_summary: ValidationCaseSummary
    validation_case_results_ref: Optional[str]
    artifacts_ref: Optional[str]
    feature_values: dict[str, Any] = field(default_factory=dict)
    feature_bin: Optional[dict[str, Any]] = None
    lineage_summary: dict[str, Any] = field(default_factory=dict)
    descendant_summary: dict[str, Any] = field(default_factory=dict)
    migration_metadata: Optional[dict[str, Any]] = None
    code_summary: str = ""
    diff_summary: Optional[str] = None
    behavior_summary: Optional[str] = None
    prior_score_history: list[dict[str, Any]] = field(default_factory=list)
    prior_ranking_events: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validation_summary(metrics: dict[str, Any], max_failures: int) -> ValidationCaseSummary:
    pass_rate = metrics.get("pass_rate")
    hidden_pass_rate = metrics.get("hidden_pass_rate")
    num_cases = int(metrics.get("num_cases", metrics.get("total_cases", 0)) or 0)
    passed = int(metrics.get("passed", metrics.get("passed_cases", 0)) or 0)
    failed = int(
        metrics.get("failed", metrics.get("failed_cases", max(0, num_cases - passed))) or 0
    )
    failures = metrics.get("notable_failures") or metrics.get("failures") or []
    if isinstance(failures, str):
        failures = [failures]
    if not isinstance(failures, list):
        failures = []
    return ValidationCaseSummary(
        num_cases=num_cases,
        passed=passed,
        failed=failed,
        pass_rate=float(pass_rate) if isinstance(pass_rate, (int, float)) else None,
        hidden_pass_rate=(
            float(hidden_pass_rate) if isinstance(hidden_pass_rate, (int, float)) else None
        ),
        notable_failures=[str(item)[:300] for item in failures[:max_failures]],
    )


def build_program_evidence(
    program, evaluation_result, database, config, iteration: int
) -> ProgramEvidence:
    metrics = dict(getattr(program, "metrics", {}) or {})
    if evaluation_result:
        metrics.update(dict(evaluation_result))
    feature_dimensions = getattr(database.config, "feature_dimensions", [])
    default_score = get_fitness_score(metrics, feature_dimensions)
    program.metrics.setdefault("_default_fitness_score", default_score)
    program.metrics.setdefault("_raw_evaluator_combined_score", metrics.get("combined_score"))
    metrics.setdefault("_default_fitness_score", program.metrics["_default_fitness_score"])
    metrics.setdefault(
        "_raw_evaluator_combined_score", program.metrics.get("_raw_evaluator_combined_score")
    )

    island = program.metadata.get("island")
    cell_key = None
    feature_bin = None
    try:
        coords = database._calculate_feature_coords(program)
        cell_key = database._feature_coords_to_key(coords)
        feature_bin = dict(zip(feature_dimensions, coords))
    except Exception:
        pass

    feature_values = {name: metrics[name] for name in feature_dimensions if name in metrics}
    code = getattr(program, "code", "") or ""
    changes = getattr(program, "changes_description", "") or program.metadata.get("changes", "")
    code_summary = changes or f"{len(code.splitlines())} lines, {len(code)} characters"
    prior_history = program.metadata.get("fitness_score_history") or []

    return ProgramEvidence(
        program_id=program.id,
        island=island,
        cell_key=cell_key,
        code_hash=hashlib.sha256(code.encode("utf-8")).hexdigest(),
        parent_program_id=program.parent_id,
        generation=program.generation,
        created_iteration=getattr(program, "iteration_found", iteration),
        raw_metrics=metrics,
        combined_score_before_rerank=metrics.get("combined_score"),
        validation_summary=_validation_summary(
            metrics, config.fitness.agentic.max_validation_cases_per_program
        ),
        validation_case_results_ref=program.metadata.get("validation_case_results_ref"),
        artifacts_ref=getattr(program, "artifact_dir", None),
        feature_values=feature_values,
        feature_bin=feature_bin,
        lineage_summary={"parent": program.parent_id, "generation": program.generation},
        migration_metadata=program.metadata.get("migration"),
        code_summary=code_summary,
        diff_summary=program.metadata.get("changes"),
        prior_score_history=list(prior_history),
    )
