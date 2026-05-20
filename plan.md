
# Agentic Fitness Research and Cross-Island Reranking Plan

Date: May 21, 2026

Repository: `algorithmicsuperintelligence/openevolve`

Reference: https://deepwiki.com/algorithmicsuperintelligence/openevolve/3.1-map-elites-algorithm

## Revision Summary

This revision changes the agentic fitness plan from mostly **single-candidate contextual scoring** into a **batch research reranking system**.

The important new requirement:

```text
When reranking a previously ranked program, do not use only its old score.
Use the program's own validation evidence, artifacts, code/context summary, lineage, island context, feature-cell context, and prior score history.
```

The fitness grader is no longer just a local scorer. It becomes a temporary research agent that does the kind of work a data scientist would do:

```text
1. Inspect the top retained programs across all islands.
2. Read compact evidence cards for every program being reranked.
3. Design one or more candidate fitness functions / scoring rubrics.
4. Apply those functions to the full reranking batch.
5. Produce a global/cross-island ranking.
6. Renormalize all programs in the scored batch.
7. Write a score epoch and preserve score provenance.
```

This preserves the original invariant:

```text
MAP-Elites still owns diversity and cell replacement.
Agentic fitness changes the score MAP-Elites compares.
```

But the score now belongs to a **score epoch**, not a one-off isolated ranking event.

## Updated Vision

OpenEvolve already has the right outer structure for self-improvement:

```text
Keep:
  MAP-Elites cells
  islands
  archive
  migration topology
  evaluator-produced raw metrics
  feature dimensions

Change:
  how fitness evidence is collected
  how program context is summarized
  how fitness functions are designed
  how combined_score is assigned
  how all retained programs are periodically renormalized
  how score epochs and score provenance are logged
  how future scoring policies improve from outcome data
```

The new conceptual model:

```text
raw evaluator metrics
  + validation traces
  + artifacts
  + program/code summary
  + parent/lineage context
  + feature-cell context
  + island context
  + historical score/outcome context
  + cross-island anchors
  -> program evidence cards
  -> fitness research agent
  -> designed fitness functions
  -> batch ranking
  -> score epoch
  -> normalized combined_score updates
  -> MAP-Elites rebuild / replacement decisions
```

This is more powerful than asking an LLM, "Is this child better than these elites?" It asks, "Given the evidence across the whole run, what scoring function should currently define usefulness, and how should all retained programs be ranked under that function?"

## Core Design Change

The old MVP path was:

```text
worker evaluates child
controller calls fitness_strategy.score_child(...)
score_child updates child.metrics["combined_score"]
controller adds rescored child to database
MAP-Elites compares target score
```

The revised research-rerank path is:

```text
worker evaluates child
controller stores child raw evaluation evidence
child enters pending scoring buffer
periodically, controller creates a cross-island rerank batch
fitness researcher receives evidence cards for every program in the batch
fitness researcher designs scoring functions
trusted code applies / validates scoring functions
all programs in the batch receive score updates for the new epoch
database updates combined_score and score history
MAP-Elites cells are rebuilt or updated using the new scores
ranking event, score epoch, and preference data are dumped
```

For low-latency operation, a hybrid mode can still do local insertion scoring. But the main research mode should treat local scores as **provisional** until the next score epoch.

## New Concept: Score Epochs

A score epoch is a coherent reranking pass over a scoped set of programs.

```python
@dataclass
class ScoreEpoch:
    epoch_id: str
    iteration: int
    scope: str
    target_islands: list[int]
    program_ids: list[str]
    fitness_function_specs: list[dict]
    ensemble_spec: dict
    normalization_spec: dict
    created_by: str
    confidence: float
    notes: str
```

Every program updated by an epoch should retain its previous score history:

```python
program.metadata.setdefault("fitness_score_history", []).append({
    "epoch_id": epoch_id,
    "previous_combined_score": old_score,
    "new_combined_score": new_score,
    "global_rank": global_rank,
    "island_rank": island_rank,
    "normalization": normalization_spec,
    "fitness_functions": fitness_function_ids,
})
```

Current score fields:

```python
metrics["_default_fitness_score"] = original_combined_score_or_average
metrics["_previous_combined_score"] = previous_combined_score
metrics["fitness_research_score"] = research_score
metrics["fitness_research_rank_global"] = global_rank
metrics["fitness_research_rank_island"] = island_rank
metrics["fitness_epoch_id"] = epoch_id
metrics["agentic_combined_score"] = normalized_score
metrics["combined_score"] = normalized_score
```

The previous score is evidence, but it must not be the only evidence.

## Program Evidence Model

Each program being reranked needs a full evidence bundle or a compact summary of one.

Add:

```text
openevolve/fitness/evidence.py
openevolve/fitness/program_card.py
openevolve/fitness/research.py
```

Core structure:

```python
@dataclass
class ProgramEvidence:
    program_id: str
    island: int
    cell_key: Optional[str]
    code_hash: str
    parent_program_id: Optional[str]
    generation: int
    created_iteration: int

    # Raw evaluator evidence
    raw_metrics: dict
    combined_score_before_rerank: Optional[float]
    validation_summary: dict
    validation_case_results_ref: Optional[str]
    artifacts_ref: Optional[str]

    # Search context
    feature_values: dict
    feature_bin: Optional[dict]
    lineage_summary: dict
    descendant_summary: dict
    migration_metadata: Optional[dict]

    # Program content
    code_summary: str
    diff_summary: Optional[str]
    behavior_summary: Optional[str]

    # Historical score context
    prior_score_history: list[dict]
    prior_ranking_events: list[str]
```

Then compress it into a prompt-safe card:

```python
@dataclass
class ProgramCard:
    program_id: str
    island: int
    cell_key: Optional[str]
    summary: str
    raw_metric_table: dict
    validation_summary: dict
    notable_failures: list[str]
    artifacts_summary: str
    lineage_summary: str
    prior_score_summary: str
```

The reranker prompt should receive `ProgramCard` objects, not just program IDs and scores.

## Fitness Researcher Output

The LLM should output structured JSON with two layers:

1. A set of proposed fitness functions.
2. The resulting ranking / score updates after applying those functions.

Required schema:

```json
{
  "research_summary": "...",
  "fitness_functions": [
    {
      "function_id": "correctness_gated_efficiency_v1",
      "purpose": "Reward programs that pass validation and improve efficiency without overfitting.",
      "inputs": ["pass_rate", "hidden_pass_rate", "speed_score", "robustness_score"],
      "formula_dsl": {
        "gate": {
          "metric": "hidden_pass_rate",
          "kind": "soft_threshold",
          "threshold": 0.98,
          "below_scale": 0.10
        },
        "components": [
          {"metric": "pass_rate", "weight": 0.35, "transform": "identity"},
          {"metric": "hidden_pass_rate", "weight": 0.30, "transform": "identity"},
          {"metric": "speed_score", "weight": 0.20, "transform": "saturating"},
          {"metric": "robustness_score", "weight": 0.15, "transform": "identity"}
        ]
      },
      "known_failure_modes": ["May under-reward novelty", "Needs hidden validation coverage"]
    }
  ],
  "ensemble": {
    "method": "weighted_sum",
    "weights": {
      "correctness_gated_efficiency_v1": 0.70,
      "novelty_promise_v1": 0.30
    }
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
      "reason": "Strong hidden validation and robust speedup."
    }
  ],
  "pairwise_preferences": [
    {
      "winner": "abc",
      "loser": "def",
      "dimension": "quality",
      "confidence": 0.86,
      "reason": "abc dominates hidden validation and robustness."
    }
  ],
  "apply_score_updates": true
}
```

## Trusted Fitness Function Application

Do not let arbitrary generated Python directly rewrite scores in production mode.

Preferred MVP design:

```text
LLM proposes DSL fitness functions.
Trusted interpreter validates the DSL.
Trusted interpreter computes scores over ProgramEvidence.
LLM can explain and rank, but score application is deterministic and auditable.
```

Reject or fall back when:

```text
function references unavailable metrics
function produces NaN / inf
normalization is undefined
all scores collapse to the same value
invalid programs outrank valid programs without stated rationale
function violates configured monotonic sanity checks
LLM ranking omits a program in the batch
LLM tries to use only previous scores
```

## Updated Config

Extend the existing `AgenticFitnessConfig` rather than replacing it.

```python
@dataclass
class AgenticFitnessConfig:
    enabled: bool = False
    model: Optional[str] = None
    temperature: float = 0.0

    # Strategy mode
    mode: str = "local_insert"  # local_insert | batch_research | hybrid

    # Local scoring, retained for compatibility / hybrid mode
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

    # Evidence / context controls
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

Example config:

```yaml
fitness:
  algo: agentic
  agentic:
    enabled: true
    mode: batch_research
    model: gpt-5.2
    research_rerank_interval: 50
    research_scope: all_retained
    research_top_k_per_island: 12
    research_include_archive: true
    research_include_pending_children: true
    research_max_programs: 128
    research_num_fitness_functions: 3
    program_context_mode: summary_plus_metrics
    normalize_scores: true
    normalization_method: rank_percentile
    rewrite_combined_score: true
    rebuild_map_elites_after_research: true
```

## Updated Controller Flow

Edit `openevolve/process_parallel.py`.

New batch-research flow:

```text
worker evaluates child
controller stores raw evaluator output and artifacts
controller creates ProgramEvidence for child
controller adds child to pending_scoring_queue

if should_run_research_rerank(iteration):
    snapshot = database.create_research_snapshot(...)
    batch = fitness_strategy.build_research_batch(snapshot, pending_scoring_queue)
    score_epoch = await fitness_strategy.run_research_rerank(batch)
    database.apply_score_epoch(score_epoch)
    database.rebuild_map_elites_if_needed()
    pending_scoring_queue.remove_scored_programs(score_epoch.program_ids)
```

Hybrid mode:

```text
child gets provisional local score
child can be inserted immediately
periodic research rerank later rewrites scores for retained programs
MAP-Elites cells are rebuilt after rewrite
```

Strict batch mode:

```text
child is not eligible to replace cell incumbents until scored in a research epoch
```

Recommended MVP default:

```text
mode: hybrid
```

Reason: hybrid keeps search moving while still letting periodic research epochs correct score drift and renormalize across islands.

## Database Changes

Edit `openevolve/database.py`.

Add methods:

```python
def create_research_snapshot(
    self,
    scope: str,
    top_k_per_island: int,
    include_archive: bool,
    include_cell_incumbents: bool,
    include_pending: bool,
) -> ResearchSnapshot:
    ...


def apply_score_epoch(self, epoch: ScoreEpoch) -> None:
    ...


def rebuild_map_elites_from_scores(self, score_key: str = "combined_score") -> None:
    ...
```

Why rebuilding matters:

```text
If combined_score changes for many programs, the current cell incumbent may no longer be the best program in that cell.
After a research rerank, MAP-Elites must be rebuilt or at least revalidated for affected cells.
```

This is the biggest semantic change from the earlier plan.

## Research Rerank Context

The batch should include:

```text
pending children not yet globally scored
current island top K programs
MAP-Elites cell incumbents
archive elites
global best programs
recent migrants
programs whose prior score is stale
same-cell competitors for high-impact cells
```

For each included program, the fitness researcher must see:

```text
program ID
island ID
feature cell / feature values
code summary or diff summary
raw evaluator metrics
validation-case summary
important validation failures
artifact summary
lineage / parent summary
descendant utility summary, if available
migration metadata, if any
prior score history, if any
```

The prior score is allowed only as historical context. It must not be the main input.

## Prompt Direction

Replace the single-candidate ranker prompt with a research prompt.

System prompt summary:

```text
You are a fitness research agent for an evolutionary program search system.
Your job is not to assign arbitrary scores.
Your job is to inspect the validation evidence across programs, design scoring functions that reflect useful downstream search pressure, apply those functions consistently across all programs in the batch, and produce a normalized cross-island ranking.

Use program validation evidence, artifacts, and context.
Do not rely only on previous scores.
Preserve correctness and robustness constraints.
Explain which fitness functions you designed and why.
Return strict JSON.
```

User prompt structure:

```text
Task context
Evaluator contract
Feature dimensions
Current MAP-Elites / island summary
Program cards
Requested output schema
Safety / validity rules
```

## Updated Logging

Keep the original files:

```text
ranking_events.jsonl
migration_events.jsonl
ranking_outcomes.jsonl
migration_outcomes.jsonl
preference_pairs.jsonl
```

Add:

```text
program_evidence.jsonl
program_cards.jsonl
fitness_research_events.jsonl
score_epochs.jsonl
score_updates.jsonl
fitness_function_specs.jsonl
```

### `program_evidence.jsonl`

```json
{
  "type": "program_evidence",
  "program_id": "abc",
  "iteration": 500,
  "island": 2,
  "cell_key": "complexity=3,speed=7",
  "raw_metrics": {"pass_rate": 1.0, "speed_score": 0.82},
  "validation_summary": {
    "num_cases": 120,
    "passed": 120,
    "failed": 0,
    "hidden_pass_rate": 0.98
  },
  "artifacts_ref": "artifacts/abc.json",
  "code_summary": "Uses cached prefix table and avoids repeated scan.",
  "lineage_summary": {"parent": "p1", "depth": 6},
  "prior_score_history": [
    {"epoch_id": "e42", "combined_score": 0.74}
  ]
}
```

### `fitness_research_events.jsonl`

```json
{
  "type": "fitness_research_event",
  "epoch_id": "epoch-0500",
  "iteration": 500,
  "scope": "all_retained",
  "target_islands": [0, 1, 2, 3],
  "program_ids": ["abc", "def", "ghi"],
  "program_card_refs": ["cards/abc.json", "cards/def.json", "cards/ghi.json"],
  "research_summary": "Correctness and hidden validation dominate; speed matters only after pass-rate gate.",
  "fitness_function_ids": ["correctness_gated_efficiency_v1", "novelty_promise_v1"],
  "normalization": {"method": "rank_percentile", "scope": "cross_island_batch"},
  "confidence": 0.83
}
```

### `score_updates.jsonl`

```json
{
  "type": "score_update",
  "epoch_id": "epoch-0500",
  "program_id": "abc",
  "previous_combined_score": 0.74,
  "new_combined_score": 0.91,
  "global_rank": 3,
  "island_rank": 1,
  "quality_score": 0.94,
  "promise_score": 0.71,
  "normalization_method": "rank_percentile",
  "reason": "High hidden pass rate and robust speedup."
}
```

## Migration Under Research Reranking

Migration still matters, but migration scoring becomes part of score-epoch logic.

For a migrant:

```text
source island score is historical evidence
target island score is recomputed from target context
research epoch can compare migrant against target island incumbents and same-cell competitors
```

Migration metadata:

```python
metadata["migration"] = {
    "source_program_id": source.id,
    "source_island": source_island,
    "target_island": target_island,
    "arrived_iteration": iteration,
    "rescored_on_arrival": True,
    "score_epoch_id": epoch_id,
}
```

The question becomes:

```text
Does this migrant add useful search pressure to the target island under the current research-designed fitness functions?
```

Not merely:

```text
Was this migrant globally good in its source island?
```

## Renormalization Requirement, Updated

Earlier plan:

```text
renorm the candidates being inserted
include enough top programs from the relevant island
avoid rewriting the whole population by default
```

Updated plan:

```text
For research-rerank mode, renormalize every program in the research batch.
By default, the batch should cover all retained high-value programs: island top K, cell incumbents, archive elites, pending children, and recent migrants.
Optionally include every evaluated historical program if configured.
```

Recommended MVP scope:

```text
all_retained = current island populations + MAP-Elites cell incumbents + archive + pending children
```

Do not default to every historical rejected candidate, because that can explode cost and context size. But preserve enough evidence so future offline research can rerank historical candidates if needed.

## Outcome Tracking

The previous outcome tracking remains, but now outcome labels should attach to score epochs and fitness function IDs.

Add questions:

```text
Which designed fitness function best predicted downstream utility?
Did the ensemble outperform the previous default score?
Which score epoch produced better descendants?
Did cross-island renormalization improve migration usefulness?
Did rewritten scores cause MAP-Elites cell churn that later proved useful or harmful?
```

Update outcome record:

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

## Tests

Add / update focused tests:

```text
tests/test_fitness_config.py
tests/test_program_evidence.py
tests/test_program_card_generation.py
tests/test_fitness_research_strategy.py
tests/test_research_rerank_events.py
tests/test_score_epoch_application.py
tests/test_map_elites_rebuild_after_score_epoch.py
tests/test_agentic_migration_rerank.py
tests/test_migration_planning.py
tests/test_research_prompt_schema.py
```

Critical assertions:

```text
default mode preserves existing combined_score behavior
batch_research mode creates ProgramEvidence for every reranked program
reranker input includes validation evidence, not only previous score
research event returns fitness function specs and rankings
all programs in a score epoch receive score_epoch_id
previous combined_score is preserved in history
combined_score is updated only after validated score update
MAP-Elites cells are rebuilt or revalidated after score rewrite
feature dimensions still control MAP-Elites placement
migration planning does not insert programs by itself
migrant target score is recalibrated in target context
ranking events and research events are valid JSONL
preference pairs can be reconstructed from score epoch rankings
outcomes can be attributed to epoch_id and fitness_function_id
```

## Implementation Order

Create a branch:

```text
agentic-fitness-research-rerank
```

Implement in this order:

```text
1. Extend fitness config with mode=batch_research/hybrid.
2. Add ProgramEvidence and ProgramCard schemas.
3. Capture raw validation data and artifacts for every evaluated program.
4. Add pending_scoring_queue for evaluated children.
5. Add deterministic ProgramCard summarizer.
6. Add FitnessResearchStrategy with deterministic stub researcher.
7. Add score epoch / score update data structures.
8. Add database.apply_score_epoch(...).
9. Add database.rebuild_map_elites_from_scores(...).
10. Add research JSONL logging.
11. Add tests for evidence completeness and score epoch updates.
12. Add LLM research prompt with strict JSON schema.
13. Add DSL fitness function interpreter and sanity checks.
14. Add migration integration with score epochs.
15. Add outcome tracking by epoch and fitness function ID.
```

## Near-Term MVP

The minimal useful version should not try to solve every scoring problem at once.

MVP behavior:

```text
hybrid mode
children receive provisional default/local scores
all retained top programs are periodically reranked in score epochs
program cards include raw metrics, validation summaries, failure summaries, artifacts summaries, lineage, and previous score history
fitness researcher proposes 1-3 DSL scoring functions
trusted interpreter applies scoring functions
score epoch rewrites combined_score for every program in the batch
MAP-Elites cells are rebuilt after rewrite
all events are dumped for offline analysis
```

This gives the desired data-scientist workflow without immediately blocking the entire search loop on large LLM calls.

## Longer-Term Direction

Once enough score epochs exist, use them as training data.

```text
fitness researcher designs scoring functions
OpenEvolve acts on those scores
outcome tracker measures downstream utility
score epoch outcomes identify useful fitness functions
preference dataset trains future ranker / fitness researcher
future ranker is evaluated on held-out tasks and problem classes
```

The meta-OpenEvolve version still makes sense, but the evolved object should now be broader:

```text
fitness_research_policy.py
```

not merely:

```text
fitness_policy.py
```

That policy should decide:

```text
which programs to include in a research batch
which evidence fields matter
which fitness functions to design
how to ensemble scoring functions
how to normalize scores
when to rewrite combined_score
how to evaluate whether an epoch was useful
```

The frozen external validation objective remains mandatory. The system can improve its fitness research policy, but it cannot redefine final success for itself.
