"""Agentic fitness research strategy with deterministic MVP reranking."""

from __future__ import annotations

import json
import logging
import uuid

from openevolve.config import LLMModelConfig
from openevolve.fitness.dsl import FitnessDSLInterpreter
from openevolve.fitness.evidence import build_program_evidence
from openevolve.fitness.normalization import normalize_scores
from openevolve.fitness.program_card import ProgramCardBuilder
from openevolve.fitness.research import FitnessFunctionSpec, ScoreEpoch, ScoreUpdate
from openevolve.llm.openai import OpenAILLM
from openevolve.utils.metrics_utils import get_fitness_score

logger = logging.getLogger(__name__)


class AgenticFitnessStrategy:
    def __init__(self, config, llm_client=None, event_writer=None):
        self.config = config
        self.llm_client = llm_client
        self.event_writer = event_writer
        self.card_builder = ProgramCardBuilder(config.fitness.agentic)
        self.dsl_interpreter = FitnessDSLInterpreter(config.fitness.agentic)
        self.research_llm = llm_client
        if self.research_llm is None and config.fitness.agentic.acp_cdc_ai_python:
            self.research_llm = self._create_acp_openai_llm()

    async def score_child(self, program, context):
        feature_dimensions = context.database.config.feature_dimensions
        default_score = get_fitness_score(program.metrics, feature_dimensions)
        program.metrics.setdefault("_default_fitness_score", default_score)
        program.metrics.setdefault(
            "_raw_evaluator_combined_score", program.metrics.get("combined_score")
        )
        program.metrics.setdefault("agentic_combined_score", default_score)
        if self.config.fitness.agentic.rewrite_combined_score:
            program.metrics.setdefault("combined_score", default_score)
        return program

    async def maybe_run_research_rerank(self, context):
        cfg = self.config.fitness.agentic
        snapshot = context.database.create_research_snapshot(
            iteration=context.iteration,
            scope=cfg.research_scope,
            top_k_per_island=cfg.research_top_k_per_island,
            include_archive=cfg.research_include_archive,
            include_cell_incumbents=cfg.research_include_cell_incumbents,
            include_pending=cfg.research_include_pending_children,
            pending_program_ids=context.pending_program_ids,
            max_programs=cfg.research_max_programs,
        )
        if not snapshot.program_ids:
            return None

        evidence = [
            build_program_evidence(program, None, context.database, self.config, context.iteration)
            for program in snapshot.programs_by_id.values()
        ]
        cards = [self.card_builder.build(item) for item in evidence]
        if self.event_writer:
            for item in evidence:
                self.event_writer.write_program_evidence(item)
            for card in cards:
                self.event_writer.write_program_card(card)

        if cfg.acp_cdc_ai_python:
            try:
                epoch = await self._run_acp_researcher(cards, snapshot, context)
                if epoch is not None:
                    if self.event_writer:
                        self.event_writer.write_fitness_research_event(epoch)
                    return epoch
            except Exception as exc:
                logger.warning(
                    "ACP fitness researcher failed; falling back to deterministic rerank: %s", exc
                )

        epoch = self._deterministic_epoch(evidence, snapshot, context)
        if self.event_writer:
            self.event_writer.write_fitness_research_event(epoch)
        return epoch

    async def score_migrant(self, migrant, migration_plan, context):
        if isinstance(migrant.metadata.get("migration"), dict):
            migrant.metadata["migration"]["rescored_on_arrival"] = False
        return await self.score_child(migrant, context)

    def _deterministic_epoch(self, evidence, snapshot, context):
        cfg = self.config.fitness.agentic
        function = FitnessFunctionSpec(
            function_id="correctness_gated_default_score_v1",
            purpose="Rank programs by validation evidence and default evaluator fitness.",
            inputs=["pass_rate", "hidden_pass_rate", "_default_fitness_score", "combined_score"],
            formula_dsl={
                "gate": {
                    "metric": "pass_rate",
                    "kind": "soft_threshold",
                    "threshold": 1.0,
                    "below_scale": 0.25,
                    "default": 1.0,
                },
                "components": [
                    {
                        "metric": "hidden_pass_rate",
                        "weight": 0.30,
                        "transform": "identity",
                        "default": 0.0,
                    },
                    {
                        "metric": "pass_rate",
                        "weight": 0.30,
                        "transform": "identity",
                        "default": 1.0,
                    },
                    {
                        "metric": "_default_fitness_score",
                        "weight": 0.40,
                        "transform": "identity",
                        "default": 0.0,
                    },
                ],
            },
            known_failure_modes=[
                "Falls back to default evaluator score when validation metrics are absent."
            ],
        )

        raw_scores = {
            item.program_id: self.dsl_interpreter.score(function, item) for item in evidence
        }
        normalized = normalize_scores(
            raw_scores, cfg.normalization_method, cfg.score_floor, cfg.score_ceiling
        )
        ordered = sorted(raw_scores, key=lambda program_id: (-raw_scores[program_id], program_id))
        island_ranks = self._island_ranks(ordered, snapshot.programs_by_id)
        epoch_id = f"fitness-epoch-{context.iteration}-{uuid.uuid4().hex[:8]}"
        updates = [
            ScoreUpdate(
                epoch_id=epoch_id,
                program_id=program_id,
                previous_combined_score=snapshot.programs_by_id[program_id].metrics.get(
                    "combined_score"
                ),
                new_combined_score=float(normalized[program_id]),
                global_rank=idx + 1,
                island_rank=island_ranks.get(program_id),
                quality_score=float(raw_scores[program_id]),
                promise_score=None,
                normalization_method=cfg.normalization_method,
                reason="Deterministic MVP research rerank.",
            )
            for idx, program_id in enumerate(ordered)
        ]
        epoch = ScoreEpoch(
            epoch_id=epoch_id,
            iteration=context.iteration,
            scope=snapshot.scope,
            target_islands=snapshot.target_islands,
            program_ids=snapshot.program_ids,
            fitness_function_specs=[function],
            ensemble_spec={"method": "weighted_sum", "weights": {function.function_id: 1.0}},
            normalization_spec={
                "method": cfg.normalization_method,
                "score_floor": cfg.score_floor,
                "score_ceiling": cfg.score_ceiling,
                "scope": "cross_island_batch",
            },
            created_by="agentic_fitness_deterministic_stub",
            confidence=0.75,
            notes="Deterministic MVP score epoch.",
            updates=updates,
        )
        if self.event_writer:
            pass
        return epoch

    def _create_acp_openai_llm(self):
        cfg = self.config.fitness.agentic
        llm_cfg = self.config.llm
        api_base = cfg.acp_cdc_ai_python_base_url.rstrip("/")
        if not api_base.endswith("/v1"):
            api_base = f"{api_base}/v1"
        return OpenAILLM(
            LLMModelConfig(
                name=cfg.model or getattr(llm_cfg, "primary_model", None) or "OPEN_AI_gpt-5.2",
                api_base=api_base,
                api_key=getattr(llm_cfg, "api_key", None) or "not-needed",
                temperature=cfg.temperature,
                top_p=getattr(llm_cfg, "top_p", None),
                max_tokens=getattr(llm_cfg, "max_tokens", 4096),
                timeout=getattr(llm_cfg, "timeout", 60),
                retries=getattr(llm_cfg, "retries", 3),
                retry_delay=getattr(llm_cfg, "retry_delay", 5),
                reasoning_effort=getattr(llm_cfg, "reasoning_effort", None),
            )
        )

    async def _run_acp_researcher(self, cards, snapshot, context):
        cfg = self.config.fitness.agentic
        cwd = cfg.acp_cdc_ai_python_cwd or context.database.config.db_path or "."
        messages = self._build_acp_messages(cards, snapshot, context, cwd)
        if self.research_llm is None:
            raise RuntimeError("ACP research LLM is not configured")

        content = await self.research_llm.generate_with_context(
            system_message=messages[0]["content"],
            messages=messages[1:],
            harness_greeting={
                "working_directory": cwd,
                "env": cfg.acp_cdc_ai_python_env,
                "mcp_servers": cfg.acp_cdc_ai_python_mcp_servers,
            },
        )
        parsed = self._parse_research_json(content, set(snapshot.program_ids))
        epoch = self._epoch_from_research_json(parsed, snapshot, context)
        return epoch

    def _build_acp_messages(self, cards, snapshot, context, cwd: str):
        card_payload = [card.to_dict() for card in cards]
        system = (
            "You are a fitness research agent for OpenEvolve. You may inspect files under "
            "the configured working directory to understand saved programs, logs, artifacts, "
            "and score history. Return strict JSON only."
        )
        user = {
            "task": "Design or select fitness functions and rerank every supplied program.",
            "working_directory": cwd,
            "iteration": context.iteration,
            "scope": snapshot.scope,
            "target_islands": snapshot.target_islands,
            "program_ids": snapshot.program_ids,
            "program_cards": card_payload,
            "required_output": {
                "research_summary": "string",
                "fitness_functions": [
                    {
                        "function_id": "string",
                        "purpose": "string",
                        "inputs": ["metric_name"],
                        "formula_dsl": {},
                        "known_failure_modes": ["string"],
                    }
                ],
                "ensemble": {"method": "weighted_sum", "weights": {"function_id": 1.0}},
                "normalization": {
                    "method": self.config.fitness.agentic.normalization_method,
                    "score_floor": self.config.fitness.agentic.score_floor,
                    "score_ceiling": self.config.fitness.agentic.score_ceiling,
                    "scope": "cross_island_batch",
                },
                "rankings": [
                    {
                        "program_id": "id",
                        "global_rank": 1,
                        "island_rank": 1,
                        "raw_research_score": 0.0,
                        "normalized_score": 1.0,
                        "quality_score": 0.0,
                        "promise_score": 0.0,
                        "confidence": 0.75,
                        "reason": "string",
                    }
                ],
                "pairwise_preferences": [],
                "apply_score_updates": True,
            },
        }
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, sort_keys=True)},
        ]

    def _parse_research_json(self, content: str, expected_program_ids: set[str]) -> dict:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end < start:
            raise ValueError("ACP researcher did not return a JSON object")
        parsed = json.loads(content[start : end + 1])
        if isinstance(parsed.get("required_output"), dict):
            parsed = parsed["required_output"]
        if not parsed.get("apply_score_updates", False):
            raise ValueError("ACP researcher declined score updates")
        ranking_ids = {item.get("program_id") for item in parsed.get("rankings", [])}
        if ranking_ids != expected_program_ids:
            missing = expected_program_ids - ranking_ids
            extra = ranking_ids - expected_program_ids
            raise ValueError(f"ACP rankings mismatch; missing={missing}, extra={extra}")
        return parsed

    def _epoch_from_research_json(self, parsed: dict, snapshot, context) -> ScoreEpoch:
        cfg = self.config.fitness.agentic
        epoch_id = f"fitness-epoch-{context.iteration}-{uuid.uuid4().hex[:8]}"
        functions = [
            FitnessFunctionSpec(
                function_id=str(item["function_id"]),
                purpose=str(item.get("purpose", "")),
                inputs=list(item.get("inputs", [])),
                formula_dsl=dict(item.get("formula_dsl", {})),
                known_failure_modes=list(item.get("known_failure_modes", [])),
            )
            for item in parsed.get("fitness_functions", [])
        ]
        if not functions:
            functions = [
                FitnessFunctionSpec(
                    function_id="acp_researcher_rank_v1",
                    purpose="ACP researcher supplied direct rankings.",
                    inputs=[],
                    formula_dsl={},
                )
            ]

        normalization = dict(parsed.get("normalization", {}))
        normalization.setdefault("method", cfg.normalization_method)
        normalization.setdefault("score_floor", cfg.score_floor)
        normalization.setdefault("score_ceiling", cfg.score_ceiling)
        normalization.setdefault("scope", "cross_island_batch")

        updates = []
        for item in sorted(parsed["rankings"], key=lambda ranking: int(ranking["global_rank"])):
            program_id = str(item["program_id"])
            updates.append(
                ScoreUpdate(
                    epoch_id=epoch_id,
                    program_id=program_id,
                    previous_combined_score=snapshot.programs_by_id[program_id].metrics.get(
                        "combined_score"
                    ),
                    new_combined_score=float(item["normalized_score"]),
                    global_rank=int(item["global_rank"]),
                    island_rank=(
                        int(item["island_rank"]) if item.get("island_rank") is not None else None
                    ),
                    quality_score=(
                        float(item["quality_score"])
                        if item.get("quality_score") is not None
                        else None
                    ),
                    promise_score=(
                        float(item["promise_score"])
                        if item.get("promise_score") is not None
                        else None
                    ),
                    normalization_method=str(normalization["method"]),
                    reason=str(item.get("reason", "")),
                )
            )

        return ScoreEpoch(
            epoch_id=epoch_id,
            iteration=context.iteration,
            scope=snapshot.scope,
            target_islands=snapshot.target_islands,
            program_ids=snapshot.program_ids,
            fitness_function_specs=functions,
            ensemble_spec=dict(parsed.get("ensemble", {})),
            normalization_spec=normalization,
            created_by="acp-cdc-ai-python",
            confidence=self._mean_confidence(parsed.get("rankings", [])),
            notes=str(parsed.get("research_summary", "")),
            updates=updates,
        )

    def _mean_confidence(self, rankings: list[dict]) -> float:
        values = [float(item.get("confidence", 0.0)) for item in rankings]
        return sum(values) / len(values) if values else 0.0

    def _island_ranks(self, ordered: list[str], programs_by_id: dict) -> dict[str, int]:
        counts: dict[int, int] = {}
        ranks: dict[str, int] = {}
        for program_id in ordered:
            island = programs_by_id[program_id].metadata.get("island")
            if island is None:
                continue
            counts[island] = counts.get(island, 0) + 1
            ranks[program_id] = counts[island]
        return ranks
