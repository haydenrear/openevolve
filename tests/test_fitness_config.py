from openevolve.config import Config


def test_default_fitness_config_is_disabled():
    config = Config()
    assert config.fitness.algo == "default"
    assert config.fitness.agentic.enabled is False


def test_agentic_fitness_config_from_dict():
    config = Config.from_dict(
        {
            "fitness": {
                "algo": "agentic",
                "agentic": {
                    "enabled": True,
                    "mode": "hybrid",
                    "research_rerank_interval": 5,
                    "research_rerank_min_pending": 1,
                },
            }
        }
    )
    assert config.fitness.algo == "agentic"
    assert config.fitness.agentic.enabled is True
    assert config.fitness.agentic.mode == "hybrid"


def test_agentic_fitness_accepts_acp_cdc_ai_python_yaml_key():
    config = Config.from_dict(
        {
            "fitness": {
                "algo": "agentic",
                "agentic": {
                    "enabled": True,
                    "acp-cdc-ai-python": True,
                    "acp_cdc_ai_python_cwd": "/tmp/research",
                },
            }
        }
    )
    assert config.fitness.agentic.acp_cdc_ai_python is True
    assert config.fitness.agentic.acp_cdc_ai_python_cwd == "/tmp/research"
