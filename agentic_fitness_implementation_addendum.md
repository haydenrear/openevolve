# Addendum: Implementation Specifics for Agentic Fitness Research Reranking

Date: May 21, 2026

Repository: `algorithmicsuperintelligence/openevolve`

Related plan: `agentic_fitness_research_rerank_plan.md`

## Purpose

The main plan defines the architecture and development direction for agentic fitness, batch research reranking, score epochs, program evidence, MAP-Elites rebuilding, migration rescoring, and outcome tracking.

This addendum converts that plan into concrete implementation details: files to add or edit, data contracts, method signatures, control flow, validation rules, logging schemas, and acceptance criteria.

The implementation target is a safe MVP that supports:

```text
hybrid agentic fitness mode
program evidence capture
program card generation
periodic cross-island research reranking
DSL-based fitness function proposals
trusted score application
score epoch history
MAP-Elites rebuild after score rewrites
migration integration
JSONL telemetry for later preference optimization
```

The critical invariant remains:

```text
MAP-Elites owns diversity and cell replacement.
Agentic fitness owns score construction and score rewriting.
```

---

## 1. Implementation Scope

### In scope for MVP

Implement the following:

```text
1. Config support for agentic batch/hybrid mode.
2. ProgramEvidence data capture after evaluation.
3. ProgramCard summarization for rerank prompts.
4. Pending scoring queue for evaluated programs.
5. FitnessResearchStrategy with deterministic stub mode.
6. Strict LLM/JSON output schema, even before real LLM integration.
7. DSL fitness function interpreter.
8. ScoreEpoch and ScoreUpdate application.
9. combined_score rewrite with score history preservation.
10. MAP-Elites rebuild/revalidation after score rewrite.
11. Migration integration with target-context score epochs.
12. JSONL event logging.
13. Tests for default compatibility, score epochs, MAP-Elites rebuild, and migration rerank.
```

### Out of scope for first MVP

Defer these until the MVP is stable:

```text
1. Training a preference model.
2. Fully evolving fitness_research_policy.py.
3. Reranking every historical rejected candidate by default.
4. Arbitrary generated Python fitness functions in production mode.
5. Using score epochs to mutate feature dimensions.
6. Replacing MAP-Elites selection logic.
7. Cross-island council as active selection pressure outside score epochs.
```

---

## 2. Files to Add

Add this package:

```text
openevolve/fitness/
  __init__.py
  strategy.py
  default.py
  agentic.py
  evidence.py
  program_card.py
  research.py
  dsl.py
  normalization.py
  events.py
  outcome_tracker.py
```

Add prompt files:

```text
openevolve/prompts/defaults/agentic_fitness_research_system.txt
openevolve/prompts/defaults/agentic_fitness_research_user.txt
openevolve/prompts/defaults/agentic_fitness_local_system.txt
openevolve/prompts/defaults/agentic_fitness_local_user.txt
```

Add tests:

```text
tests/test_fitness_config.py
tests/test_program_evidence.py
tests/test_program_card_generation.py
tests/test_fitness_research_strategy.py
tests/test_fitness_dsl.py
tests/test_research_rerank_events.py
tests/test_score_epoch_application.py
tests/test_map_elites_rebuild_after_score_epoch.py
tests/test_agentic_migration_rerank.py
tests/test_migration_planning.py
tests/test_research_prompt_schema.py
tests/test_agentic_default_compatibility.py
```

---

## 3. Files to Edit

Edit these existing files:

```text
openevolve/config.py
openevolve/process_parallel.py
openevolve/database.py
openevolve/evaluator.py or equivalent evaluation result path
openevolve/models.py or equivalent Program model definition
openevolve/utils/metrics.py or equivalent get_fitness_score location
```

Exact names may vary with the current repository layout. The implementation rule is:

```text
Do not put LLM scoring inside ProgramDatabase.
ProgramDatabase may create snapshots, apply score epochs, and rebuild MAP-Elites.
The controller/process layer decides when to call the fitness strategy.
```

---

## 4. Config Contract

Add or extend config objects in `openevolve/config.py`.

```python
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FitnessConfig:
    algo: str = "default"  # default | agentic
    agentic: "AgenticFitnessConfig" = field(default_factory=lambda: AgenticFitnessConfig())


@dataclass
class AgenticFitnessConfig:
    enabled: bool = False
    model: Optional[str] = None
    temperature: float = 0.0

    # Strategy mode
    mode: str = "local_insert"  # local_insert | batch_research | hybrid

    # Local/provisional scoring
    ranking_pool_top_k: int = 8
    archive_top_k: int = 4
    quality_weight: float = 0.65
    promise_weight: float = 0.25
    default_fitness_weight: float = 0.10

    # Batch research reranking
    research_rerank_interval: int = 50
    research_rerank_min_pending: int = 16
    research_scope: str = "all_retained"  # top_per_island | all_retained | all_evaluated
    research_top_k_per_island: int = 12
    research_include_archive: bool = True
    research_include_cell_incumbents: bool = True
    research_include_pending_children: bool = True
    research_include_migrants: bool = True
    research_max_programs: int = 128
    research_num_fitness_functions: int = 3
    research_function_language: str = "dsl"  # dsl | python_sandboxed

    # Evidence/context controls
    program_context_mode: str = "summary_plus_metrics"  # summary | summary_plus_metrics | full
    program_card_token_budget: int = 700
    max_validation_cases_per_program: int = 20
    include_validation_failure_examples: bool = True
    include_artifact_summaries: bool = True
    include_lineage_summaries: bool = True
    include_prior_score_history: bool = True

    # Score update policy
    normalize_scores: bool = True
    normalization_method: str = "rank_percentile"  # rank_percentile | zscore_sigmoid | minmax
    score_floor: float = 0.0
    score_ceiling: float = 1.0
    apply_research_scores: bool = True
    rewrite_combined_score: bool = True
    preserve_score_history: bool = True
    rebuild_map_elites_after_research: bool = True

    # Migration scoring
    rescore_migrants: bool = True
    migration_context_top_k: int = 8
    migration_source_top_k: int = 4
    migration_target_top_k: int = 8
    migration_include_cell_incumbent: bool = True

    # Logging
    dump_ranking_events: bool = True
    dump_research_events: bool = True
    ranking_events_path: Optional[str] = None
    research_events_path: Optional[str] = None
```

Validation rules:

```text
fitness.algo must be one of: default, agentic
agentic.mode must be one of: local_insert, batch_research, hybrid
research_max_programs must be >= 1
score_floor must be less than score_ceiling
weights must be finite and non-negative
normalization_method must be supported
research_function_language must be dsl for production MVP
```

---

## 5. Strategy Interface

Create `openevolve/fitness/strategy.py`.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Optional


@dataclass
class FitnessContext:
    iteration: int
    run_id: str
    task_name: Optional[str]
    problem_class: Optional[str]
    target_island: Optional[int]
    database: object
    config: object


class FitnessStrategy(Protocol):
    async def score_child(self, program, context: FitnessContext):
        ...

    async def maybe_run_research_rerank(self, context: FitnessContext):
        ...

    async def score_migrant(self, migrant, migration_plan, context: FitnessContext):
        ...
```

Create factory:

```python
def create_fitness_strategy(config, llm_client=None, event_writer=None):
    if getattr(config.fitness, "algo", "default") == "agentic" and config.fitness.agentic.enabled:
        return AgenticFitnessStrategy(config, llm_client=llm_client, event_writer=event_writer)
    return DefaultFitnessStrategy(config)
```

`DefaultFitnessStrategy` must be a no-op for scoring:

```python
class DefaultFitnessStrategy:
    async def score_child(self, program, context):
        return program

    async def maybe_run_research_rerank(self, context):
        return None

    async def score_migrant(self, migrant, migration_plan, context):
        return migrant
```

Acceptance criterion:

```text
With fitness.algo=default, program metrics and MAP-Elites behavior are byte-for-byte or semantically unchanged except for harmless metadata absence/presence explicitly allowed by tests.
```

---

## 6. Data Models

Create `openevolve/fitness/evidence.py`.

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class ValidationCaseSummary:
    num_cases: int = 0
    passed: int = 0
    failed: int = 0
    pass_rate: Optional[float] = None
    hidden_num_cases: Optional[int] = None
    hidden_passed: Optional[int] = None
    hidden_failed: Optional[int] = None
    hidden_pass_rate: Optional[float] = None
    notable_failures: list[str] = field(default_factory=list)


@dataclass
class ProgramEvidence:
    program_id: str
    island: Optional[int]
    cell_key: Optional[str]
    code_hash: Optional[str]
    parent_program_id: Optional[str]
    generation: Optional[int]
    created_iteration: Optional[int]

    raw_metrics: dict[str, Any]
    combined_score_before_rerank: Optional[float]
    validation_summary: ValidationCaseSummary
    validation_case_results_ref: Optional[str]
    artifacts_ref: Optional[str]

    feature_values: dict[str, Any] = field(default_factory=dict)
    feature_bin: Optional[dict[str, Any]] = None
    lineage_summary: dict[str, Any] = field(default_factory=dict)
    descendant_summary: dict[str, Any] = field(default_factory=dict)
    migration_metadata: Optional[dict[str, Any]] = None

    code_summary: str = ""
    diff_summary: Optional[str] = None
    behavior_summary: Optional[str] = None

    prior_score_history: list[dict[str, Any]] = field(default_factory=list)
    prior_ranking_events: list[str] = field(default_factory=list)
```

Create `openevolve/fitness/program_card.py`.

```python
@dataclass
class ProgramCard:
    program_id: str
    island: Optional[int]
    cell_key: Optional[str]
    summary: str
    raw_metric_table: dict
    validation_summary: dict
    notable_failures: list[str]
    artifacts_summary: str
    lineage_summary: str
    prior_score_summary: str
```

Create `openevolve/fitness/research.py`.

```python
@dataclass
class FitnessFunctionSpec:
    function_id: str
    purpose: str
    inputs: list[str]
    formula_dsl: dict
    known_failure_modes: list[str] = field(default_factory=list)


@dataclass
class ProgramRanking:
    program_id: str
    global_rank: int
    island_rank: Optional[int]
    raw_research_score: float
    normalized_score: float
    quality_score: Optional[float]
    promise_score: Optional[float]
    confidence: float
    reason: str


@dataclass
class ScoreUpdate:
    epoch_id: str
    program_id: str
    previous_combined_score: Optional[float]
    new_combined_score: float
    global_rank: int
    island_rank: Optional[int]
    quality_score: Optional[float]
    promise_score: Optional[float]
    normalization_method: str
    reason: str


@dataclass
class ScoreEpoch:
    epoch_id: str
    iteration: int
    scope: str
    target_islands: list[int]
    program_ids: list[str]
    fitness_function_specs: list[FitnessFunctionSpec]
    ensemble_spec: dict
    normalization_spec: dict
    created_by: str
    confidence: float
    notes: str
    updates: list[ScoreUpdate] = field(default_factory=list)
```

---

## 7. Evidence Capture

Evidence should be captured immediately after evaluator completion and before any score rewrite.

Add helper:

```python
def build_program_evidence(program, evaluation_result, database, config, iteration) -> ProgramEvidence:
    ...
```

Rules:

```text
1. Store raw evaluator metrics exactly as returned.
2. Preserve the original combined_score or default average under _default_fitness_score.
3. Store validation case details by reference if too large.
4. Summarize artifacts; do not inline huge artifacts into ProgramCard.
5. Include prior score history if this is a retained/reranked program.
6. Include lineage, parent ID, island, feature values, feature bin, and migration metadata when available.
7. Never let prior score replace validation evidence.
```

Recommended metric preservation:

```python
original = program.metrics.get("combined_score")
if original is None:
    original = compute_default_fitness_score(program.metrics, config)

program.metrics.setdefault("_default_fitness_score", original)
program.metrics.setdefault("_raw_evaluator_combined_score", program.metrics.get("combined_score"))
```

---

## 8. Program Card Generation

Program cards are prompt-safe summaries of evidence.

Create:

```python
class ProgramCardBuilder:
    def __init__(self, config):
        self.config = config

    def build(self, evidence: ProgramEvidence) -> ProgramCard:
        ...
```

Card construction rules:

```text
1. Always include program_id, island, cell_key, raw metrics, validation summary, and prior score summary.
2. Include at most max_validation_cases_per_program failures/examples.
3. Truncate code_summary, artifacts_summary, and lineage_summary to fit program_card_token_budget.
4. Mark omitted fields explicitly, e.g. "artifacts_summary_truncated": true.
5. Never include full source code by default; include code summary and diff summary.
```

Minimum card payload:

```json
{
  "program_id": "abc",
  "island": 2,
  "cell_key": "complexity=3,speed=7",
  "summary": "Uses cached prefix table and avoids repeated scan.",
  "raw_metric_table": {
    "pass_rate": 1.0,
    "speed_score": 0.82,
    "robustness_score": 0.76,
    "_default_fitness_score": 0.74
  },
  "validation_summary": {
    "num_cases": 120,
    "passed": 120,
    "failed": 0,
    "hidden_pass_rate": 0.98
  },
  "notable_failures": [],
  "artifacts_summary": "No anomalous artifacts.",
  "lineage_summary": "Parent p1, depth 6, two useful descendants.",
  "prior_score_summary": "Previous epoch e42 score 0.74; no earlier rewrites."
}
```

---

## 9. Pending Scoring Queue

Add a controller-owned queue, not a database-owned queue.

```python
class PendingScoringQueue:
    def __init__(self):
        self._program_ids: list[str] = []
        self._seen: set[str] = set()

    def add(self, program_id: str) -> None:
        if program_id not in self._seen:
            self._program_ids.append(program_id)
            self._seen.add(program_id)

    def remove_many(self, program_ids: set[str]) -> None:
        self._program_ids = [p for p in self._program_ids if p not in program_ids]
        self._seen -= program_ids

    def __len__(self) -> int:
        return len(self._program_ids)
```

Rules:

```text
1. Add every evaluated child to the queue after evidence capture.
2. In hybrid mode, the child may also be inserted immediately with provisional score.
3. In strict batch_research mode, the child is not eligible for MAP-Elites replacement until a score epoch assigns a score.
4. Remove IDs only after successful score epoch application.
```

---

## 10. Controller Integration

Edit `openevolve/process_parallel.py`.

Add initialization:

```python
self.fitness_strategy = create_fitness_strategy(
    self.config,
    llm_client=self.llm_client,
    event_writer=self.event_writer,
)
self.pending_scoring_queue = PendingScoringQueue()
```

Child evaluation path:

```python
async def _handle_evaluated_child(self, child, evaluation_result, iteration):
    evidence = build_program_evidence(
        program=child,
        evaluation_result=evaluation_result,
        database=self.database,
        config=self.config,
        iteration=iteration,
    )
    child.metadata["program_evidence"] = evidence_to_ref_or_dict(evidence)
    self.event_writer.write_program_evidence(evidence)

    mode = self.config.fitness.agentic.mode

    if self.config.fitness.algo == "agentic" and mode in {"local_insert", "hybrid"}:
        child = await self.fitness_strategy.score_child(
            child,
            FitnessContext(
                iteration=iteration,
                run_id=self.run_id,
                task_name=self.task_name,
                problem_class=self.problem_class,
                target_island=getattr(child, "island", None),
                database=self.database,
                config=self.config,
            ),
        )

    if self.config.fitness.algo != "agentic" or mode in {"local_insert", "hybrid"}:
        self.database.add(child)

    if self.config.fitness.algo == "agentic" and mode in {"batch_research", "hybrid"}:
        self.pending_scoring_queue.add(child.id)

    await self._maybe_run_research_rerank(iteration)
```

Research trigger:

```python
def should_run_research_rerank(self, iteration: int) -> bool:
    cfg = self.config.fitness.agentic
    if self.config.fitness.algo != "agentic" or not cfg.enabled:
        return False
    if cfg.mode not in {"batch_research", "hybrid"}:
        return False
    if len(self.pending_scoring_queue) < cfg.research_rerank_min_pending:
        return False
    return iteration % cfg.research_rerank_interval == 0
```

Research execution:

```python
async def _maybe_run_research_rerank(self, iteration: int):
    if not self.should_run_research_rerank(iteration):
        return

    context = FitnessContext(
        iteration=iteration,
        run_id=self.run_id,
        task_name=self.task_name,
        problem_class=self.problem_class,
        target_island=None,
        database=self.database,
        config=self.config,
    )

    epoch = await self.fitness_strategy.maybe_run_research_rerank(context)
    if epoch is None:
        return

    self.database.apply_score_epoch(epoch)

    if self.config.fitness.agentic.rebuild_map_elites_after_research:
        self.database.rebuild_map_elites_from_scores(score_key="combined_score")

    self.pending_scoring_queue.remove_many(set(epoch.program_ids))
```

---

## 11. Database Snapshot and Score Application

Edit `openevolve/database.py`.

Add:

```python
@dataclass
class ResearchSnapshot:
    iteration: int
    scope: str
    target_islands: list[int]
    program_ids: list[str]
    programs_by_id: dict[str, object]
    island_top_program_ids: dict[int, list[str]]
    cell_incumbent_ids: list[str]
    archive_program_ids: list[str]
    pending_program_ids: list[str]
```

Add methods:

```python
def create_research_snapshot(
    self,
    scope: str,
    top_k_per_island: int,
    include_archive: bool,
    include_cell_incumbents: bool,
    include_pending: bool,
    pending_program_ids: Optional[list[str]] = None,
    max_programs: Optional[int] = None,
) -> ResearchSnapshot:
    ...
```

Selection priority for `all_retained`:

```text
1. Pending children.
2. Current island best programs.
3. MAP-Elites cell incumbents.
4. Archive elites.
5. Recent migrants.
6. Stale-score programs.
7. Same-cell competitors for high-impact cells.
```

Add:

```python
def apply_score_epoch(self, epoch: ScoreEpoch) -> None:
    for update in epoch.updates:
        program = self.get_program(update.program_id)
        if program is None:
            continue

        old_score = program.metrics.get("combined_score")

        program.metadata.setdefault("fitness_score_history", []).append({
            "epoch_id": epoch.epoch_id,
            "previous_combined_score": old_score,
            "new_combined_score": update.new_combined_score,
            "global_rank": update.global_rank,
            "island_rank": update.island_rank,
            "normalization": epoch.normalization_spec,
            "fitness_function_ids": [f.function_id for f in epoch.fitness_function_specs],
        })

        program.metrics["_previous_combined_score"] = old_score
        program.metrics["fitness_research_score"] = update.new_combined_score
        program.metrics["fitness_research_rank_global"] = update.global_rank
        if update.island_rank is not None:
            program.metrics["fitness_research_rank_island"] = update.island_rank
        program.metrics["fitness_epoch_id"] = epoch.epoch_id
        program.metrics["agentic_combined_score"] = update.new_combined_score
        program.metrics["combined_score"] = update.new_combined_score
```

Add:

```python
def rebuild_map_elites_from_scores(self, score_key: str = "combined_score") -> None:
    # Clear all cell incumbents.
    # Reinsert retained programs by feature cell.
    # Within each cell, keep the program with max metrics[score_key].
    # Preserve feature dimension/bin logic exactly as before.
    ...
```

Rules:

```text
1. Rebuild only from retained/active programs, not every historical rejected program unless configured.
2. Do not mutate feature values during rebuild.
3. If two programs tie on score, use deterministic tie-break: higher _default_fitness_score, then earlier created_iteration, then program_id lexical order.
4. After rebuild, archive/global best caches must be refreshed.
```

---

## 12. Fitness Research Strategy

Create `openevolve/fitness/agentic.py`.

```python
class AgenticFitnessStrategy:
    def __init__(self, config, llm_client=None, event_writer=None):
        self.config = config
        self.llm_client = llm_client
        self.event_writer = event_writer
        self.card_builder = ProgramCardBuilder(config.fitness.agentic)
        self.dsl_interpreter = FitnessDSLInterpreter(config.fitness.agentic)

    async def score_child(self, program, context):
        # Local provisional scoring path.
        # Use deterministic stub or small local ranker.
        # Preserve default score and set agentic_combined_score.
        return program

    async def maybe_run_research_rerank(self, context):
        cfg = self.config.fitness.agentic
        snapshot = context.database.create_research_snapshot(
            scope=cfg.research_scope,
            top_k_per_island=cfg.research_top_k_per_island,
            include_archive=cfg.research_include_archive,
            include_cell_incumbents=cfg.research_include_cell_incumbents,
            include_pending=cfg.research_include_pending_children,
            pending_program_ids=getattr(context, "pending_program_ids", None),
            max_programs=cfg.research_max_programs,
        )

        evidence = [self._load_or_build_evidence(p, context) for p in snapshot.programs_by_id.values()]
        cards = [self.card_builder.build(e) for e in evidence]

        raw_result = await self._call_researcher(cards, snapshot, context)
        parsed = parse_and_validate_research_result(raw_result, expected_program_ids=snapshot.program_ids)

        epoch = self._build_score_epoch(parsed, evidence, snapshot, context)
        self._validate_epoch(epoch, evidence)
        return epoch
```

Deterministic stub researcher:

```text
Purpose: make tests pass without LLM.
Behavior:
  - create one correctness_gated_default_score DSL function
  - score each program from raw metrics
  - rank by score descending
  - normalize by rank_percentile
  - produce deterministic pairwise preferences among adjacent ranks
```

---

## 13. DSL Fitness Function Interpreter

Create `openevolve/fitness/dsl.py`.

Supported DSL:

```json
{
  "gate": {
    "metric": "hidden_pass_rate",
    "kind": "soft_threshold",
    "threshold": 0.98,
    "below_scale": 0.10
  },
  "components": [
    {"metric": "pass_rate", "weight": 0.35, "transform": "identity"},
    {"metric": "speed_score", "weight": 0.20, "transform": "saturating"},
    {"metric": "robustness_score", "weight": 0.15, "transform": "identity"}
  ],
  "penalties": [
    {"metric": "timeout_rate", "weight": 0.20, "transform": "identity"}
  ]
}
```

Supported transforms:

```text
identity: clamp metric into [0, 1]
saturating: 1 - exp(-max(x, 0))
log1p: log(1 + max(x, 0)) / log(2), clamped
inverse: 1 / (1 + max(x, 0))
capped: min(max(x, 0), cap)
boolean: 1.0 if truthy else 0.0
```

Supported gates:

```text
none
hard_threshold: 0 if metric < threshold else 1
soft_threshold: below_scale * metric/threshold if below threshold else 1
linear: clamp metric into [0, 1]
```

Interpreter sketch:

```python
class FitnessDSLInterpreter:
    def score(self, spec: FitnessFunctionSpec, evidence: ProgramEvidence) -> float:
        metrics = flatten_metrics(evidence)
        gate = self._compute_gate(spec.formula_dsl.get("gate"), metrics)
        total = 0.0
        for component in spec.formula_dsl.get("components", []):
            value = self._metric_value(metrics, component["metric"])
            total += float(component.get("weight", 1.0)) * self._transform(value, component)
        for penalty in spec.formula_dsl.get("penalties", []):
            value = self._metric_value(metrics, penalty["metric"])
            total -= float(penalty.get("weight", 1.0)) * self._transform(value, penalty)
        return clamp_finite(gate * total, 0.0, 1.0)
```

Validation rules:

```text
1. All referenced metrics must exist or have an explicit default.
2. All weights must be finite.
3. Score must be finite for every program.
4. A function cannot use only previous scores.
5. Invalid programs cannot outrank fully valid programs unless the function explicitly declares novelty/exploration purpose and the ensemble weight is below a configured cap.
6. At least one validation/correctness metric must be referenced when available.
```

---

## 14. Normalization

Create `openevolve/fitness/normalization.py`.

Supported methods:

```text
rank_percentile
minmax
zscore_sigmoid
```

Recommended MVP default:

```text
rank_percentile
```

Rank percentile:

```python
def rank_percentile(scores: dict[str, float], floor: float, ceiling: float) -> dict[str, float]:
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    n = len(ordered)
    if n == 1:
        return {ordered[0][0]: ceiling}
    out = {}
    for idx, (program_id, _) in enumerate(ordered):
        pct = 1.0 - idx / (n - 1)
        out[program_id] = floor + pct * (ceiling - floor)
    return out
```

Rules:

```text
1. Normalize within the score epoch scope.
2. Store normalization scope and method in ScoreEpoch.
3. Do not mix local-island normalization with cross-island normalization without marking the scope.
4. When using island-specific rank, still store global normalized score if combined_score is global.
```

---

## 15. LLM Research Output Schema

The prompt must require strict JSON. The parser must reject missing programs.

Required output:

```json
{
  "research_summary": "string",
  "fitness_functions": [
    {
      "function_id": "string",
      "purpose": "string",
      "inputs": ["metric_name"],
      "formula_dsl": {},
      "known_failure_modes": ["string"]
    }
  ],
  "ensemble": {
    "method": "weighted_sum",
    "weights": {"function_id": 1.0}
  },
  "normalization": {
    "method": "rank_percentile",
    "score_floor": 0.0,
    "score_ceiling": 1.0,
    "scope": "cross_island_batch"
  },
  "rankings": [
    {
      "program_id": "abc",
      "global_rank": 1,
      "island_rank": 1,
      "raw_research_score": 0.923,
      "normalized_score": 0.997,
      "quality_score": 0.95,
      "promise_score": 0.77,
      "confidence": 0.86,
      "reason": "string"
    }
  ],
  "pairwise_preferences": [
    {
      "winner": "abc",
      "loser": "def",
      "dimension": "quality",
      "confidence": 0.86,
      "reason": "string"
    }
  ],
  "apply_score_updates": true
}
```

Parser fallback policy:

```text
invalid JSON -> no epoch; keep current scores
missing program -> no epoch unless allow_partial_epoch=true
NaN/inf/out-of-range -> clamp if safe, else no epoch
unknown DSL transform -> reject function
unknown metric reference -> reject function
all scores equal -> keep rank order by default score or reject epoch
low confidence -> optionally apply blend with default score, but mark epoch confidence low
```

---

## 16. Score Epoch Application Semantics

When applying an epoch:

```text
1. Validate that every update references an active program.
2. Validate finite score in configured range.
3. Preserve prior combined_score.
4. Append metadata fitness_score_history entry.
5. Set metrics["agentic_combined_score"].
6. Set metrics["combined_score"] only if rewrite_combined_score=true.
7. Set metrics["fitness_epoch_id"].
8. Write score_updates.jsonl.
9. Rebuild or revalidate MAP-Elites affected cells.
```

Important distinction:

```text
Score epoch application changes selection pressure.
It does not change feature dimensions or feature cell placement.
```

---

## 17. MAP-Elites Rebuild Details

After rewriting scores for multiple programs, existing cell incumbents may be stale.

Rebuild algorithm:

```text
1. Clear current MAP-Elites cell map.
2. Iterate over retained active programs.
3. Compute each program's cell key from existing feature dimensions.
4. Compare program against current cell incumbent using get_fitness_score or score_key.
5. Store best program per cell.
6. Refresh island/global best caches.
7. Log number of changed incumbents.
```

Do not:

```text
1. Recompute feature dimensions unless the evaluator has changed them.
2. Re-evaluate source code.
3. Drop archive entries solely because their score changed.
4. Run LLM calls inside rebuild.
```

Acceptance criterion:

```text
If program A and B occupy the same cell and B receives a higher combined_score in a score epoch, rebuild makes B the cell incumbent.
```

---

## 18. Migration Integration

Migration planning remains separate from insertion.

Existing plan:

```python
@dataclass
class MigrationPlan:
    source_program_id: str
    source_island: int
    target_island: int
    reason: str = "scheduled_migration"
```

Add metadata:

```python
metadata["migration"] = {
    "source_program_id": source.id,
    "source_island": source_island,
    "target_island": target_island,
    "arrived_iteration": iteration,
    "rescored_on_arrival": False,
    "score_epoch_id": None,
}
```

In hybrid mode:

```text
1. Migrant is copied into target island with source score preserved as historical evidence.
2. Migrant enters pending_scoring_queue.
3. Next score epoch reranks migrant against target context.
4. After score update, metadata.rescored_on_arrival=true and score_epoch_id is set.
```

In immediate migrant rescore mode:

```text
1. Build a small target-context research batch.
2. Include migrant, target top K, source top K, same-cell incumbent, archive anchors.
3. Apply score epoch to the migrant and any included target programs.
4. Insert/rebuild target MAP-Elites after score application.
```

Acceptance criterion:

```text
A migrant's source-island score is never used as the final target-island combined_score unless agentic rescoring is disabled or fallback occurs.
```

---

## 19. Event Logging

Implement `openevolve/fitness/events.py`.

```python
class FitnessEventWriter:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def write_jsonl(self, filename: str, record: dict) -> None:
        ...

    def write_program_evidence(self, evidence: ProgramEvidence) -> None:
        ...

    def write_program_card(self, card: ProgramCard) -> None:
        ...

    def write_fitness_research_event(self, epoch: ScoreEpoch, extra: dict | None = None) -> None:
        ...

    def write_score_update(self, update: ScoreUpdate) -> None:
        ...
```

Files:

```text
program_evidence.jsonl
program_cards.jsonl
fitness_research_events.jsonl
score_epochs.jsonl
score_updates.jsonl
fitness_function_specs.jsonl
ranking_events.jsonl
migration_events.jsonl
ranking_outcomes.jsonl
migration_outcomes.jsonl
preference_pairs.jsonl
```

Durability rules:

```text
1. JSONL writes should be append-only.
2. Each record must include run_id, iteration, and timestamp.
3. Each score update must include epoch_id.
4. Each fitness function spec must include function_id and epoch_id.
5. If a score epoch fails validation, write a rejected_research_event record with reason.
```

---

## 20. Outcome Tracking

Create `openevolve/fitness/outcome_tracker.py`.

```python
@dataclass
class OutcomeHorizon:
    horizon: int
    label: str


class ScoreEpochOutcomeTracker:
    def __init__(self, horizons: list[int]):
        self.horizons = horizons
        self.pending_epoch_ids: dict[str, int] = {}

    def register_epoch(self, epoch: ScoreEpoch):
        self.pending_epoch_ids[epoch.epoch_id] = epoch.iteration

    def collect_due_outcomes(self, current_iteration: int, database) -> list[dict]:
        ...
```

Outcome record:

```json
{
  "type": "score_epoch_outcome",
  "epoch_id": "epoch-0500",
  "origin_iteration": 500,
  "current_iteration": 550,
  "horizon": 50,
  "fitness_function_ids": ["correctness_gated_efficiency_v1"],
  "num_programs_updated": 96,
  "num_cell_incumbents_changed": 14,
  "best_score_delta": 0.13,
  "best_validation_delta": 0.09,
  "descendant_improvement_rate": 0.21,
  "migration_success_rate_delta": 0.04
}
```

MVP can collect only:

```text
survives_in_population
survives_in_archive
became_cell_incumbent
became_island_best
best_descendant_score
num_descendants
```

---

## 21. Prompt Implementation

System prompt should say:

```text
You are a fitness research agent for an evolutionary program search system.
Inspect program evidence across islands.
Design scoring functions that reflect useful downstream search pressure.
Use validation data, artifacts, raw metrics, lineage, migration context, and prior score history.
Prior scores are historical context only.
Do not rank using only prior scores.
Return strict JSON matching the schema.
```

User prompt sections:

```text
1. Task context
2. Evaluator contract
3. Feature dimensions
4. Current island/MAP-Elites summary
5. Normalization requirements
6. Program cards
7. Required output schema
8. Validity and fallback rules
```

Hard instruction:

```text
Every program_id in the batch must appear exactly once in rankings.
```

---

## 22. Concurrency and Async Rules

```text
1. Only the controller should trigger score epochs.
2. Do not let workers concurrently mutate database scores.
3. Score epoch application must be atomic with respect to MAP-Elites rebuild.
4. If using multiprocessing, serialize epoch application through the same controller path that handles database.add.
5. If an epoch fails halfway, rollback or leave all scores unchanged.
```

Suggested pattern:

```python
with self.database.write_lock():
    self.database.apply_score_epoch(epoch)
    self.database.rebuild_map_elites_from_scores()
```

If no lock abstraction exists, keep epoch application in the single controller event loop.

---

## 23. Fallback Behavior

Fallbacks must preserve search progress.

```text
LLM unavailable:
  use deterministic stub or skip epoch

invalid research JSON:
  reject epoch, keep current scores, write rejected event

invalid DSL:
  reject bad function; if no valid functions remain, reject epoch

missing validation evidence:
  use raw metrics and mark evidence incomplete

all normalized scores collapse:
  reject epoch or rank by default score with low confidence

MAP-Elites rebuild fails:
  restore previous cell map if snapshot exists, or stop with explicit error in tests
```

No fallback should silently rewrite `combined_score` without score provenance.

---

## 24. Test Plan

### Config tests

Assert:

```text
Default config uses fitness.algo=default.
Agentic config parses mode=hybrid and mode=batch_research.
Invalid mode raises config validation error.
Invalid normalization method raises config validation error.
```

### Evidence tests

Assert:

```text
ProgramEvidence includes raw metrics.
ProgramEvidence includes validation summary.
ProgramEvidence includes prior score history for reranked programs.
ProgramEvidence does not collapse to previous score only.
```

### Card tests

Assert:

```text
ProgramCard includes program_id, raw metrics, validation summary, prior score summary.
Large artifacts/code summaries are truncated.
Failure examples obey max_validation_cases_per_program.
```

### DSL tests

Assert:

```text
identity, saturating, capped, inverse transforms produce finite scores.
soft_threshold gate penalizes below-threshold correctness.
Unknown metric causes validation failure unless default is configured.
NaN/inf input is clamped or rejected.
Function using only previous score is rejected.
```

### Score epoch tests

Assert:

```text
apply_score_epoch preserves previous combined_score.
apply_score_epoch writes fitness_epoch_id.
apply_score_epoch updates combined_score only when rewrite_combined_score=true.
Every update is logged.
```

### MAP-Elites rebuild tests

Assert:

```text
After score rewrite, same-cell incumbent changes to highest-scoring program.
Feature cell placement is unchanged.
Tie-break is deterministic.
Global/island best caches refresh.
```

### Migration tests

Assert:

```text
MigrationPlan does not insert by itself.
Migrant metadata includes source and target island.
Agentic mode queues migrant for target-context reranking.
Target score differs from source score when epoch updates it.
Fallback keeps source score but marks rescored_on_arrival=false.
```

### Default compatibility tests

Assert:

```text
With fitness.algo=default, child insertion behavior is unchanged.
With fitness.algo=default, migration behavior is unchanged.
With fitness.algo=default, no score_epoch_id is written.
```

---

## 25. Development Order With Acceptance Gates

### Phase 1: Safe scaffolding

Implement:

```text
config
fitness package
DefaultFitnessStrategy
AgenticFitnessStrategy shell
FitnessEventWriter shell
```

Acceptance:

```text
All existing tests pass.
Default mode unchanged.
```

### Phase 2: Evidence and cards

Implement:

```text
ProgramEvidence
ProgramCard
ProgramCardBuilder
program_evidence.jsonl
program_cards.jsonl
```

Acceptance:

```text
Every evaluated program can produce evidence and a card.
No score rewrites yet.
```

### Phase 3: Deterministic score epochs

Implement:

```text
ScoreEpoch
ScoreUpdate
stub researcher
DSL interpreter
normalization
apply_score_epoch
```

Acceptance:

```text
A deterministic epoch can rerank a small batch and rewrite combined_score with history.
```

### Phase 4: MAP-Elites rebuild

Implement:

```text
rebuild_map_elites_from_scores
changed-incumbent logging
cache refresh
```

Acceptance:

```text
Cells reflect rewritten scores after epoch application.
```

### Phase 5: Controller hook

Implement:

```text
pending_scoring_queue
hybrid mode child path
research rerank trigger
score epoch application from controller
```

Acceptance:

```text
A run can evaluate children, insert provisionally, periodically rerank, and rebuild cells.
```

### Phase 6: Migration integration

Implement:

```text
MigrationPlan refactor
migrant metadata
migrant pending queue integration
target-context scoring batch
```

Acceptance:

```text
Migrants are rescored or queued for rescoring in target context.
```

### Phase 7: Real LLM research prompt

Implement:

```text
research system/user prompts
strict JSON parser
schema validation
fallback paths
```

Acceptance:

```text
LLM output can be validated, rejected, or applied deterministically.
```

### Phase 8: Outcome tracker

Implement:

```text
score_epoch_outcome records
horizon collection
preference pair extraction from rankings
```

Acceptance:

```text
Score epochs can be linked to downstream utility labels.
```

---

## 26. Definition of Done

The MVP is done when:

```text
1. Default OpenEvolve behavior is preserved under default config.
2. Agentic hybrid mode runs end-to-end without blocking search.
3. Every reranked program has ProgramEvidence or an explicit incomplete-evidence marker.
4. Research rerank batches include cross-island context and validation summaries.
5. The research output includes fitness function specs and rankings.
6. Trusted code, not arbitrary LLM Python, applies score updates.
7. Every score rewrite records epoch_id and previous score history.
8. MAP-Elites is rebuilt or revalidated after multi-program score rewrites.
9. Migrants are scored in target-island context or clearly marked as fallback.
10. JSONL logs are sufficient to reconstruct ranking decisions and score epochs.
11. Tests cover default compatibility, evidence completeness, score epochs, DSL validation, MAP-Elites rebuild, and migration reranking.
```

---

## 27. One-Sentence Implementation Summary

Implement agentic fitness as a controller-driven score-epoch system: collect evidence, summarize programs, ask a research agent to design DSL fitness functions, apply validated scores deterministically, preserve score provenance, rebuild MAP-Elites, and log enough data to train future fitness researchers.
