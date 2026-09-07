from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl

from .schema import Schema, numeric_columns


ID_LIKE = {"pair_id", "hand_id", "player_a", "player_b", "table_id", "phase", "behavior", "y_pair", "y_evidence"}


def canonical_pairs(df: pl.DataFrame, schema: Schema) -> pl.DataFrame:
    return (
        df.select(
            pl.col(schema.pair_id).alias("pair_id"),
            pl.col(schema.pair_player_1).alias("p1_raw"),
            pl.col(schema.pair_player_2).alias("p2_raw"),
        )
        .with_columns(
            pl.when(pl.col("p1_raw") <= pl.col("p2_raw")).then(pl.col("p1_raw")).otherwise(pl.col("p2_raw")).alias("player_a"),
            pl.when(pl.col("p1_raw") <= pl.col("p2_raw")).then(pl.col("p2_raw")).otherwise(pl.col("p1_raw")).alias("player_b"),
        )
        .drop("p1_raw", "p2_raw")
        .unique("pair_id")
    )


def build_pair_hand_index(seats_path: Path, pairs: pl.DataFrame, schema: Schema) -> pl.DataFrame:
    """Generate only shared hands for requested pairs."""
    hid, pid = schema.hand_id, schema.player_id
    s = pl.scan_parquet(seats_path).select([hid, pid])
    left = s.rename({pid: "player_a_raw"})
    right = s.rename({pid: "player_b_raw"})
    combos = (
        left.join(right, on=hid, how="inner")
        .filter(pl.col("player_a_raw") < pl.col("player_b_raw"))
        .select(
            pl.col(hid).alias("hand_id"),
            pl.col("player_a_raw").alias("player_a"),
            pl.col("player_b_raw").alias("player_b"),
        )
    )
    pair_lookup = pairs.lazy().select(["pair_id", "player_a", "player_b"])
    return (
        combos.join(pair_lookup, on=["player_a", "player_b"], how="inner")
        .select(["pair_id", "hand_id", "player_a", "player_b"])
        .collect(engine="streaming")
    )


def _safe_numeric_columns(path: Path, exclude: set[str]) -> list[str]:
    return numeric_columns(path, exclude)


def build_action_player_hand_features(actions_path: Path, schema: Schema, action_types: Iterable[str]) -> pl.DataFrame:
    hid, pid = schema.hand_id, schema.player_id
    exclude = {hid, pid}
    if schema.action_order:
        exclude.add(schema.action_order)
    numeric = _safe_numeric_columns(actions_path, exclude)

    lf = pl.scan_parquet(actions_path)
    exprs: list[pl.Expr] = [pl.len().alias("act_n")]
    for c in numeric:
        exprs.extend([
            pl.col(c).mean().alias(f"act_{c}_mean"),
            pl.col(c).max().alias(f"act_{c}_max"),
            pl.col(c).sum().alias(f"act_{c}_sum"),
            pl.col(c).std().fill_null(0).alias(f"act_{c}_std"),
        ])

    if schema.action_type and schema.action_type in lf.collect_schema().names():
        action_lower = pl.col(schema.action_type).cast(pl.Utf8).str.to_lowercase()
        exprs.append(action_lower.n_unique().alias("act_type_nunique"))
        for action in action_types:
            normalized = action.lower()
            safe = normalized.replace("-", "_").replace(" ", "_")
            exprs.append((action_lower == normalized).sum().alias(f"act_count_{safe}"))

    return (
        lf.group_by([hid, pid])
        .agg(exprs)
        .rename({hid: "hand_id", pid: "player_id"})
        .collect(engine="streaming")
    )


def attach_hands(pair_hands: pl.DataFrame, hands_path: Path, schema: Schema) -> pl.DataFrame:
    cols = pl.scan_parquet(hands_path).collect_schema().names()
    select_exprs = [pl.col(schema.hand_id).alias("hand_id")]
    if schema.table_id and schema.table_id in cols:
        select_exprs.append(pl.col(schema.table_id).alias("table_id"))
    if schema.phase and schema.phase in cols:
        select_exprs.append(pl.col(schema.phase).alias("phase"))

    numeric = _safe_numeric_columns(hands_path, {schema.hand_id})
    for c in numeric:
        if c not in {schema.table_id}:
            select_exprs.append(pl.col(c).alias(f"hand_{c}"))

    hands = pl.scan_parquet(hands_path).select(select_exprs).collect(engine="streaming")
    return pair_hands.join(hands, on="hand_id", how="left")


def attach_seat_features(pair_hands: pl.DataFrame, seats_path: Path, schema: Schema) -> pl.DataFrame:
    hid, pid = schema.hand_id, schema.player_id
    numeric = _safe_numeric_columns(seats_path, {hid, pid})
    seats = (
        pl.scan_parquet(seats_path)
        .select([pl.col(hid).alias("hand_id"), pl.col(pid).alias("player_id")] + [pl.col(c) for c in numeric])
        .collect(engine="streaming")
    )

    a = seats.rename({"player_id": "player_a", **{c: f"seat_a_{c}" for c in numeric}})
    b = seats.rename({"player_id": "player_b", **{c: f"seat_b_{c}" for c in numeric}})
    out = pair_hands.join(a, on=["hand_id", "player_a"], how="left").join(b, on=["hand_id", "player_b"], how="left")

    derived = []
    for c in numeric:
        ca, cb = f"seat_a_{c}", f"seat_b_{c}"
        derived.extend([
            (pl.col(ca) + pl.col(cb)).alias(f"seat_pair_{c}_sum"),
            (pl.col(ca) - pl.col(cb)).abs().alias(f"seat_pair_{c}_absdiff"),
        ])
    return out.with_columns(derived) if derived else out


def attach_action_features(pair_hands: pl.DataFrame, action_ph: pl.DataFrame) -> pl.DataFrame:
    feature_cols = [c for c in action_ph.columns if c not in {"hand_id", "player_id"}]
    a = action_ph.rename({"player_id": "player_a", **{c: f"a_{c}" for c in feature_cols}})
    b = action_ph.rename({"player_id": "player_b", **{c: f"b_{c}" for c in feature_cols}})
    out = pair_hands.join(a, on=["hand_id", "player_a"], how="left").join(b, on=["hand_id", "player_b"], how="left")

    derived = []
    for c in feature_cols:
        ca, cb = f"a_{c}", f"b_{c}"
        if out.schema.get(ca, pl.String).is_numeric() and out.schema.get(cb, pl.String).is_numeric():
            derived.extend([
                (pl.col(ca).fill_null(0) + pl.col(cb).fill_null(0)).alias(f"pair_{c}_sum"),
                (pl.col(ca).fill_null(0) - pl.col(cb).fill_null(0)).abs().alias(f"pair_{c}_absdiff"),
            ])
    return out.with_columns(derived) if derived else out


def build_pair_hand_features(
    pair_hands: pl.DataFrame,
    hands_path: Path,
    seats_path: Path,
    actions_path: Path,
    schema: Schema,
    action_types: Iterable[str],
) -> pl.DataFrame:
    out = attach_hands(pair_hands, hands_path, schema)
    out = attach_seat_features(out, seats_path, schema)
    action_ph = build_action_player_hand_features(actions_path, schema, action_types)
    out = attach_action_features(out, action_ph)
    return out


def model_feature_columns(df: pl.DataFrame, extra_exclude: Iterable[str] = ()) -> list[str]:
    exclude = ID_LIKE | set(extra_exclude)
    cols = []
    for c, dtype in df.schema.items():
        if c in exclude:
            continue
        if dtype.is_numeric() or dtype == pl.Boolean:
            cols.append(c)
    return cols


def _topk_expr(score_col: str, k: int) -> pl.Expr:
    return pl.col(score_col).sort(descending=True).head(k).mean().alias(f"evidence_top{k}_mean")


def aggregate_pair_features(pair_hand: pl.DataFrame, score_col: str = "evidence_score", top_k=(1, 2, 3, 5, 10)) -> pl.DataFrame:
    numeric = model_feature_columns(pair_hand, extra_exclude={score_col})
    agg: list[pl.Expr] = [
        pl.len().alias("n_shared_hands"),
        pl.col(score_col).max().alias("evidence_max"),
        pl.col(score_col).mean().alias("evidence_mean"),
        pl.col(score_col).std().fill_null(0).alias("evidence_std"),
        pl.col(score_col).quantile(0.95).alias("evidence_p95"),
        pl.col(score_col).quantile(0.99).alias("evidence_p99"),
        (pl.col(score_col) >= 0.5).sum().alias("evidence_count_ge_050"),
        (pl.col(score_col) >= 0.8).sum().alias("evidence_count_ge_080"),
    ]
    for k in top_k:
        agg.append(_topk_expr(score_col, k))
    for c in numeric:
        agg.extend([
            pl.col(c).mean().alias(f"ph_{c}_mean"),
            pl.col(c).max().alias(f"ph_{c}_max"),
        ])
    if "table_id" in pair_hand.columns:
        agg.append(pl.col("table_id").first().alias("table_id"))

    out = pair_hand.group_by("pair_id").agg(agg)
    return out.with_columns([
        (pl.col("evidence_max") / (1.0 + pl.col("n_shared_hands").cast(pl.Float64).log1p())).alias("evidence_max_log_exposure"),
        (pl.col("evidence_top5_mean") * pl.col("n_shared_hands").cast(pl.Float64).sqrt().clip(1, 30)).alias("evidence_top5_sqrt_exposure")
        if "evidence_top5_mean" in out.columns else pl.lit(0.0).alias("evidence_top5_sqrt_exposure"),
    ])


def rank_normalize(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return (ranks + 0.5) / max(len(values), 1)
