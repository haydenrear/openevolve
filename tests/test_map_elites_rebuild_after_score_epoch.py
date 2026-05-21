from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase


def test_rebuild_map_elites_promotes_higher_rewritten_score_same_cell():
    config = Config()
    config.database.feature_dimensions = ["score"]
    config.database.feature_bins = 1
    db = ProgramDatabase(config.database)
    low = Program(id="low", code="def f(): return 1", metrics={"score": 0.5, "combined_score": 0.2})
    high = Program(
        id="high", code="def g(): return 2", metrics={"score": 0.5, "combined_score": 0.9}
    )
    db.add(low, target_island=0)
    db.add(high, target_island=0)
    db.rebuild_map_elites_from_scores()
    assert list(db.island_feature_maps[0].values()) == ["high"]
