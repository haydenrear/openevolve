"""Append-only event writer for agentic fitness telemetry."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, is_dataclass
from typing import Any


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


class FitnessEventWriter:
    def __init__(self, base_dir: str, run_id: str | None = None):
        self.base_dir = base_dir
        self.run_id = run_id
        os.makedirs(base_dir, exist_ok=True)

    def write_jsonl(self, filename: str, record: dict[str, Any]) -> None:
        payload = {
            "timestamp": time.time(),
            "run_id": self.run_id,
            **_to_jsonable(record),
        }
        with open(os.path.join(self.base_dir, filename), "a") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")

    def write_program_evidence(self, evidence) -> None:
        self.write_jsonl(
            "program_evidence.jsonl", {"type": "program_evidence", **evidence.to_dict()}
        )

    def write_program_card(self, card) -> None:
        self.write_jsonl("program_cards.jsonl", {"type": "program_card", **card.to_dict()})

    def write_fitness_research_event(self, epoch, extra: dict[str, Any] | None = None) -> None:
        self.write_jsonl(
            "fitness_research_events.jsonl",
            {"type": "score_epoch", "epoch": epoch, **(extra or {})},
        )

    def write_score_update(self, update) -> None:
        self.write_jsonl("score_updates.jsonl", {"type": "score_update", "update": update})
