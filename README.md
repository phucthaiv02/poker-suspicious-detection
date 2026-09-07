# Poker Collusion Detection — evidence-first baseline

A reproducible baseline for the Kaggle challenge **Detect Suspicious Value Transfers in Poker**. The project treats the task as **positive-unlabelled pair ranking + multiple-instance evidence retrieval** rather than a plain tabular classifier.

> Scope: this repository is for synthetic-data anomaly/fraud detection research. It does not implement poker-playing strategy or wagering advice.

## What the pipeline does

1. **Schema inspection** — validates IDs and automatically detects common column names.
2. **Pair-hand construction** — builds shared hands only for development-labelled and evaluation pairs.
3. **Action/player-hand aggregation** — converts ordered action logs into numeric behavioural summaries while retaining `hand_id`.
4. **Evidence model** — trains on planted evidence hands as positives and hands from confirmed non-target pairs as clean negatives. Unlisted/unknown pairs are never silently treated as negatives.
5. **OOF stacking** — evidence scores used by the pair model are generated out-of-fold to avoid stacking leakage.
6. **Pair aggregation** — top-k evidence score statistics, exposure features, and pair-hand behavioural aggregates.
7. **Pair risk model** — produces the ranking score used for Pair AP.
8. **Behavior heads** — three one-vs-rest models for the disclosed families. High-risk / low-known-confidence pairs are routed to `other_coordination` as an open-set reject option.
9. **Evidence retrieval** — ranks evaluation hands by evidence score and submits the top five unique shared hand IDs.

## Repository layout

```text
configs/baseline.yaml          model + path configuration
scripts/inspect_schema.py      inspect actual competition columns
scripts/build_features.py      build dev/eval pair-hand feature caches
scripts/train.py               OOF evidence + pair + behavior training
scripts/predict.py             generate submission.csv
scripts/run_all.py             end-to-end command
src/poker_collusion/           reusable package
tests/                         lightweight metric/submission tests
artifacts/                     generated feature/model caches (gitignored)
data/                          Kaggle competition files (gitignored)
```

## Setup

Python 3.10+ is recommended.

```bash
python -m pip install -e .
```

Place the competition files under `data/`:

```text
data/
  players.parquet
  hands.parquet
  seats.parquet
  actions.parquet
  development_labels.csv
  development_evidence.csv
  evaluation_pairs.csv
  sample_submission.csv
```

Do **not** commit competition data to GitHub.

## First command: inspect the real schema

Because the public description does not enumerate every physical column name, run:

```bash
python scripts/inspect_schema.py --config configs/baseline.yaml
```

The code automatically detects common naming conventions. If a column differs, add an override in `configs/baseline.yaml`, for example:

```yaml
schema_overrides:
  pair_player_1: player_left_id
  pair_player_2: player_right_id
  action_type: action
  action_order: action_index
```

## Run the baseline

```bash
python scripts/run_all.py --config configs/baseline.yaml --output submission.csv
```

Or step by step:

```bash
python scripts/build_features.py --config configs/baseline.yaml
python scripts/train.py --config configs/baseline.yaml
python scripts/predict.py --config configs/baseline.yaml --output submission.csv
```

## Why the evidence model uses only clean negatives

`development_labels.csv` is PU-style: an unlisted pair is **unknown**, not negative. The evidence trainer therefore uses:

- positive: rows listed in `development_evidence.csv`;
- negative: shared hands belonging to confirmed non-target labelled pairs;
- positive-pair hands not listed as evidence: scored, but not forced to negative during evidence fitting.

This avoids a common label-noise mistake.

## Leakage controls

The evidence model is stacked into the pair model. Using in-sample evidence predictions would leak labels. The training code generates held-out scores with `GroupKFold`, preferring `table_id`/pool when available and falling back to `pair_id`.

Raw `player_id`, `pair_id`, and `table_id` are not used as numeric model features. They are used for joins/grouping only.

## Metric-aware behavior routing

The competition accepts only one `predicted_behavior`. The code trains three disclosed-family heads and tunes two OOF thresholds:

- `none_threshold`: low-risk pairs become `none`;
- `known_confidence`: high-risk pairs that fit none of the three disclosed families become `other_coordination`.

This treats the undisclosed fourth family as an open-set detection problem rather than inventing fake supervised labels.

## Evidence ranking

For every evaluation pair, the pipeline ranks its shared evaluation hands by evidence probability and emits up to five unique IDs. `NO_EVIDENCE` is used only when fewer than five shared hands are available.

## Recommended next upgrades

The baseline deliberately avoids overcomplicating the first version. High-value upgrades after validating the actual schema:

- pair-vs-player-history residual features;
- temporal burst/change-point features inside the evaluation period;
- behavior-specific evidence heads;
- empirical exposure correction using confirmed negative pairs;
- action-sequence motifs for coordinated isolation;
- PU bagging / calibrated pseudo-negatives;
- LightGBM + CatBoost rank ensemble.

## Tests

```bash
pytest -q
```

## Notes

The exact Kaggle metric notebook should remain the final source of truth. `src/poker_collusion/metrics.py` contains development helpers for Pair AP, disclosed Behavior MAP, and textual MAP@5 behaviour, but does not invent an overall component weighting that is not stated in the competition description.
