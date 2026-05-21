from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase
from openevolve.fitness.research import FitnessFunctionSpec, ScoreEpoch, ScoreUpdate


def test_apply_score_epoch_rewrites_combined_score_and_history():
    config = Config()
    db = ProgramDatabase(config.database)
    db.add(Program(id="p1", code="def f(): return 1", metrics={"combined_score": 0.2}))
    epoch = ScoreEpoch(
        epoch_id="e1",
        iteration=1,
        scope="all_retained",
        target_islands=[0],
        program_ids=["p1"],
        fitness_function_specs=[FitnessFunctionSpec("f", "purpose", [], {})],
        ensemble_spec={},
        normalization_spec={"method": "rank_percentile"},
        created_by="test",
        confidence=1.0,
        notes="",
        updates=[ScoreUpdate("e1", "p1", 0.2, 0.9, 1, 1, None, None, "rank_percentile", "test")],
    )
    db.apply_score_epoch(epoch)
    program = db.get("p1")
    assert program.metrics["combined_score"] == 0.9
    assert program.metadata["fitness_score_history"][0]["previous_combined_score"] == 0.2
