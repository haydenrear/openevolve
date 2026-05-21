"""Data models for agentic fitness research reranking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class FitnessFunctionSpec:
    function_id: str
    purpose: str
    inputs: list[str]
    formula_dsl: dict[str, Any]
    known_failure_modes: list[str] = field(default_factory=list)


@dataclass
class ProgramRanking:
    program_id: str
    global_rank: int
    island_rank: Optional[int]
    raw_research_score: float
    normalized_score: float
    quality_score: Optional[float]
    promise_score: Optional[float]
    confidence: float
    reason: str


@dataclass
class ScoreUpdate:
    epoch_id: str
    program_id: str
    previous_combined_score: Optional[float]
    new_combined_score: float
    global_rank: int
    island_rank: Optional[int]
    quality_score: Optional[float]
    promise_score: Optional[float]
    normalization_method: str
    reason: str


@dataclass
class ScoreEpoch:
    epoch_id: str
    iteration: int
    scope: str
    target_islands: list[int]
    program_ids: list[str]
    fitness_function_specs: list[FitnessFunctionSpec]
    ensemble_spec: dict[str, Any]
    normalization_spec: dict[str, Any]
    created_by: str
    confidence: float
    notes: str
    updates: list[ScoreUpdate] = field(default_factory=list)


@dataclass
class ResearchSnapshot:
    iteration: int
    scope: str
    target_islands: list[int]
    program_ids: list[str]
    programs_by_id: dict[str, Any]
    island_top_program_ids: dict[int, list[str]]
    cell_incumbent_ids: list[str]
    archive_program_ids: list[str]
    pending_program_ids: list[str]
