from openevolve.config import Config
from openevolve.fitness.evidence import ProgramEvidence, ValidationCaseSummary
from openevolve.fitness.program_card import ProgramCardBuilder


def test_program_card_contains_required_sections():
    config = Config()
    evidence = ProgramEvidence(
        program_id="p1",
        island=1,
        cell_key="1-2",
        code_hash=None,
        parent_program_id="p0",
        generation=1,
        created_iteration=2,
        raw_metrics={"combined_score": 0.5},
        combined_score_before_rerank=0.5,
        validation_summary=ValidationCaseSummary(num_cases=1, passed=1, pass_rate=1.0),
        validation_case_results_ref=None,
        artifacts_ref=None,
        code_summary="small change",
    )
    card = ProgramCardBuilder(config.fitness.agentic).build(evidence)
    assert card.program_id == "p1"
    assert card.raw_metric_table["combined_score"] == 0.5
    assert card.validation_summary["pass_rate"] == 1.0
