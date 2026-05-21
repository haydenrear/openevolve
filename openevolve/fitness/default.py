"""Default no-op fitness strategy."""

from __future__ import annotations


class DefaultFitnessStrategy:
    def __init__(self, config):
        self.config = config

    async def score_child(self, program, context):
        return program

    async def maybe_run_research_rerank(self, context):
        return None

    async def score_migrant(self, migrant, migration_plan, context):
        return migrant
