from openevolve.fitness.dsl import FitnessDSLInterpreter
from openevolve.fitness.evidence import ProgramEvidence, ValidationCaseSummary
from openevolve.fitness.research import FitnessFunctionSpec


def test_fitness_dsl_scores_validation_and_default_metrics():
    evidence = ProgramEvidence(
        program_id="p1",
        island=0,
        cell_key="0-0",
        code_hash=None,
        parent_program_id=None,
        generation=0,
        created_iteration=0,
        raw_metrics={"_default_fitness_score": 0.8, "pass_rate": 1.0},
        combined_score_before_rerank=0.8,
        validation_summary=ValidationCaseSummary(pass_rate=1.0),
        validation_case_results_ref=None,
        artifacts_ref=None,
    )
    spec = FitnessFunctionSpec(
        function_id="f",
        purpose="test",
        inputs=["pass_rate", "_default_fitness_score"],
        formula_dsl={
            "gate": {"metric": "pass_rate", "kind": "hard_threshold", "threshold": 1.0},
            "components": [
                {"metric": "pass_rate", "weight": 0.5, "transform": "identity"},
                {"metric": "_default_fitness_score", "weight": 0.5, "transform": "identity"},
            ],
        },
    )
    assert FitnessDSLInterpreter().score(spec, evidence) == 0.9
