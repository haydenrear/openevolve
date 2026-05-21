import pytest

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase
from openevolve.fitness.agentic import AgenticFitnessStrategy
from openevolve.fitness.strategy import FitnessContext


@pytest.mark.asyncio
async def test_agentic_strategy_builds_deterministic_epoch():
    config = Config()
    config.fitness.algo = "agentic"
    config.fitness.agentic.enabled = True
    config.fitness.agentic.research_rerank_min_pending = 1
    db = ProgramDatabase(config.database)
    db.add(
        Program(
            id="p1", code="def f(): return 1", metrics={"combined_score": 0.2, "pass_rate": 1.0}
        )
    )
    db.add(
        Program(
            id="p2", code="def g(): return 2", metrics={"combined_score": 0.8, "pass_rate": 1.0}
        )
    )
    strategy = AgenticFitnessStrategy(config)
    epoch = await strategy.maybe_run_research_rerank(
        FitnessContext(1, "run", None, None, None, db, config, ["p1", "p2"])
    )
    assert epoch is not None
    assert {update.program_id for update in epoch.updates} == {"p1", "p2"}
    assert epoch.updates[0].program_id == "p2"
