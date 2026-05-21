"""Minimal outcome tracking scaffold for future score epoch analysis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OutcomeHorizon:
    iterations: int


class ScoreEpochOutcomeTracker:
    def __init__(self):
        self._epochs = []

    def register_epoch(self, epoch) -> None:
        self._epochs.append(epoch)

    def collect_due_outcomes(self, current_iteration: int) -> list[dict]:
        return []
