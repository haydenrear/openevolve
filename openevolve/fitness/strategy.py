"""Fitness strategy interface and factory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass
class FitnessContext:
    iteration: int
    run_id: str
    task_name: Optional[str]
    problem_class: Optional[str]
    target_island: Optional[int]
    database: object
    config: object
    pending_program_ids: Optional[list[str]] = None


class FitnessStrategy(Protocol):
    async def score_child(self, program, context: FitnessContext): ...

    async def maybe_run_research_rerank(self, context: FitnessContext): ...

    async def score_migrant(self, migrant, migration_plan, context: FitnessContext): ...


def create_fitness_strategy(config, llm_client=None, event_writer=None):
    if getattr(config.fitness, "algo", "default") == "agentic" and config.fitness.agentic.enabled:
        from openevolve.fitness.agentic import AgenticFitnessStrategy

        return AgenticFitnessStrategy(config, llm_client=llm_client, event_writer=event_writer)

    from openevolve.fitness.default import DefaultFitnessStrategy

    return DefaultFitnessStrategy(config)
