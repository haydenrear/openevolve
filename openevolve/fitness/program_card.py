"""Prompt-safe program cards for fitness research."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass
class ProgramCard:
    program_id: str
    island: Optional[int]
    cell_key: Optional[str]
    summary: str
    raw_metric_table: dict[str, Any]
    validation_summary: dict[str, Any]
    notable_failures: list[str]
    artifacts_summary: str
    lineage_summary: str
    prior_score_summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProgramCardBuilder:
    def __init__(self, config):
        self.config = config

    def build(self, evidence) -> ProgramCard:
        budget = max(120, int(getattr(self.config, "program_card_token_budget", 700)))
        max_chars = budget * 4
        raw_metrics = {
            key: value
            for key, value in evidence.raw_metrics.items()
            if isinstance(value, (int, float, str, bool)) or value is None
        }
        prior = evidence.prior_score_history[-3:]
        prior_summary = "No prior score epochs."
        if prior:
            prior_summary = "; ".join(
                f"{item.get('epoch_id')} {item.get('previous_combined_score')}->{item.get('new_combined_score')}"
                for item in prior
            )
        artifacts_summary = "No artifact summary."
        if evidence.artifacts_ref:
            artifacts_summary = f"Artifacts stored at {evidence.artifacts_ref}"
        summary = (evidence.code_summary or "No code summary.")[:max_chars]
        return ProgramCard(
            program_id=evidence.program_id,
            island=evidence.island,
            cell_key=evidence.cell_key,
            summary=summary,
            raw_metric_table=raw_metrics,
            validation_summary=asdict(evidence.validation_summary),
            notable_failures=evidence.validation_summary.notable_failures,
            artifacts_summary=artifacts_summary[:max_chars],
            lineage_summary=str(evidence.lineage_summary)[:max_chars],
            prior_score_summary=prior_summary[:max_chars],
        )
