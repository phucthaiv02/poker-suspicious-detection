from __future__ import annotations

import polars as pl

from .schema import Schema, TARGET_BEHAVIORS


def normalize_development_labels(labels: pl.DataFrame, schema: Schema) -> pl.DataFrame:
    """Return canonical pair_id, y_pair, behavior plus original pair columns.

    Unlisted pairs are intentionally absent and therefore never converted to negatives.
    """
    df = labels
    behavior_expr = None
    if schema.behavior and schema.behavior in df.columns:
        behavior_expr = pl.col(schema.behavior).cast(pl.Utf8).str.to_lowercase()

    if schema.label and schema.label in df.columns:
        dtype = df.schema[schema.label]
        if dtype == pl.Boolean:
            y_expr = pl.col(schema.label).cast(pl.Int8)
        elif dtype.is_numeric():
            y_expr = (pl.col(schema.label).cast(pl.Float64) > 0).cast(pl.Int8)
        else:
            val = pl.col(schema.label).cast(pl.Utf8).str.to_lowercase()
            y_expr = val.is_in(["1", "true", "positive", "target", "coordinated", "yes"]).cast(pl.Int8)
    elif behavior_expr is not None:
        y_expr = behavior_expr.is_in(list(TARGET_BEHAVIORS)).cast(pl.Int8)
    else:
        raise ValueError("Need either a label or behavior column in development_labels.csv")

    out = df.with_columns(y_expr.alias("y_pair"))
    if behavior_expr is not None:
        out = out.with_columns(
            pl.when(pl.col("y_pair") == 1)
            .then(behavior_expr)
            .otherwise(pl.lit("none"))
            .alias("behavior")
        )
    else:
        out = out.with_columns(
            pl.when(pl.col("y_pair") == 1)
            .then(pl.lit("unknown_target"))
            .otherwise(pl.lit("none"))
            .alias("behavior")
        )
    return out


def canonical_evidence(evidence: pl.DataFrame, schema: Schema) -> pl.DataFrame:
    pair_col = schema.pair_id if schema.pair_id in evidence.columns else "pair_id"
    return evidence.select(
        pl.col(pair_col).alias("pair_id"),
        pl.col(schema.evidence_hand).alias("hand_id"),
    ).unique()
