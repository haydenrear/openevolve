from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase
from openevolve.fitness.evidence import build_program_evidence


def test_program_evidence_preserves_default_score_and_validation_summary():
    config = Config()
    db = ProgramDatabase(config.database)
    program = Program(
        id="p1",
        code="def f(): return 1",
        metrics={"combined_score": 0.7, "pass_rate": 1.0, "num_cases": 10, "passed": 10},
    )
    evidence = build_program_evidence(program, program.metrics, db, config, iteration=3)
    assert evidence.program_id == "p1"
    assert evidence.combined_score_before_rerank == 0.7
    assert evidence.validation_summary.pass_rate == 1.0
    assert program.metrics["_default_fitness_score"] == 0.7
