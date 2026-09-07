from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score

from .config import resolve_paths
from .features import (
    aggregate_pair_features,
    build_pair_hand_features,
    build_pair_hand_index,
    canonical_pairs,
    model_feature_columns,
    rank_normalize,
)
from .io import read, write_parquet
from .labels import canonical_evidence, normalize_development_labels
from .metrics import behavior_map
from .models import fit_binary_oof, fit_full_binary
from .schema import DISCLOSED_BEHAVIORS, detect_schema
from .submission import build_submission, choose_behaviors, top_evidence


def _to_pandas_features(df: pl.DataFrame, features: list[str]) -> pd.DataFrame:
    return df.select(features).fill_nan(None).fill_null(0).to_pandas()


def _pair_groups(df: pl.DataFrame) -> np.ndarray:
    if "table_id" in df.columns:
        return df["table_id"].cast(pl.Utf8).to_numpy()
    return df["pair_id"].cast(pl.Utf8).to_numpy()


def prepare_features(cfg: dict) -> dict[str, Path]:
    paths = resolve_paths(cfg)
    schema = detect_schema(paths, cfg.get("schema_overrides", {}))
    artifacts = paths["artifacts_dir"]
    (artifacts / "schema.json").write_text(json.dumps(schema.to_dict(), indent=2), encoding="utf-8")

    labels_raw = read(paths["development_labels"])
    labels = normalize_development_labels(labels_raw, schema)
    eval_raw = read(paths["evaluation_pairs"])

    dev_pairs = canonical_pairs(labels, schema)
    eval_pairs = canonical_pairs(eval_raw, schema)

    dev_index = build_pair_hand_index(paths["seats"], dev_pairs, schema)
    eval_index = build_pair_hand_index(paths["seats"], eval_pairs, schema)

    action_types = cfg["features"].get("action_types", [])
    dev_ph = build_pair_hand_features(dev_index, paths["hands"], paths["seats"], paths["actions"], schema, action_types)
    eval_ph = build_pair_hand_features(eval_index, paths["hands"], paths["seats"], paths["actions"], schema, action_types)

    if "phase" in dev_ph.columns:
        dev_ph = dev_ph.filter(pl.col("phase").cast(pl.Utf8).str.to_lowercase() == "development")
        eval_ph = eval_ph.filter(pl.col("phase").cast(pl.Utf8).str.to_lowercase() == "evaluation")

    labels_canon = labels.select([
        pl.col(schema.pair_id).alias("pair_id"),
        "y_pair",
        "behavior",
    ] + ([pl.col(schema.table_id).alias("label_table_id")] if schema.table_id and schema.table_id in labels.columns else []))
    dev_ph = dev_ph.join(labels_canon, on="pair_id", how="left")

    evidence = canonical_evidence(read(paths["development_evidence"]), schema)
    dev_ph = dev_ph.join(evidence.with_columns(pl.lit(1).alias("y_evidence")), on=["pair_id", "hand_id"], how="left")
    dev_ph = dev_ph.with_columns(pl.col("y_evidence").fill_null(0).cast(pl.Int8))

    write_parquet(dev_ph, artifacts / "dev_pair_hand.parquet")
    write_parquet(eval_ph, artifacts / "eval_pair_hand.parquet")
    labels.write_parquet(artifacts / "development_labels_normalized.parquet")
    evidence.write_parquet(artifacts / "development_evidence_normalized.parquet")
    eval_pairs.write_parquet(artifacts / "evaluation_pairs_canonical.parquet")
    return {
        "dev_pair_hand": artifacts / "dev_pair_hand.parquet",
        "eval_pair_hand": artifacts / "eval_pair_hand.parquet",
    }


def train(cfg: dict) -> dict:
    paths = resolve_paths(cfg)
    art = paths["artifacts_dir"]
    dev = pl.read_parquet(art / "dev_pair_hand.parquet")
    eval_ph = pl.read_parquet(art / "eval_pair_hand.parquet")

    seed = int(cfg["seed"])
    n_splits = int(cfg["n_splits"])

    evidence_features = model_feature_columns(dev)
    train_mask = ((dev["y_pair"] == 0) | (dev["y_evidence"] == 1)).to_numpy()
    clean = dev.filter(pl.Series(train_mask))
    X_clean = _to_pandas_features(clean, evidence_features)
    y_clean = clean["y_evidence"].to_numpy()
    groups_clean = _pair_groups(clean)

    X_all_dev = _to_pandas_features(dev, evidence_features)
    groups_all_dev = _pair_groups(dev)
    evidence_oof_result, all_dev_oof = fit_binary_oof(
        X_clean,
        y_clean,
        groups_clean,
        cfg["models"]["evidence"],
        n_splits,
        seed,
        predict_X=X_all_dev,
        predict_groups=groups_all_dev,
    )
    if np.isnan(all_dev_oof).any():
        fallback = fit_full_binary(X_clean, y_clean, cfg["models"]["evidence"], seed + 999)
        missing = np.isnan(all_dev_oof)
        all_dev_oof[missing] = fallback.predict_proba(X_all_dev.loc[missing])[:, 1]

    dev_scored = dev.with_columns(pl.Series("evidence_score", all_dev_oof))
    eval_X = _to_pandas_features(eval_ph, evidence_features)
    full_evidence = fit_full_binary(X_clean, y_clean, cfg["models"]["evidence"], seed)
    eval_evidence_score = full_evidence.predict_proba(eval_X)[:, 1]
    eval_scored = eval_ph.with_columns(pl.Series("evidence_score", eval_evidence_score))

    write_parquet(dev_scored, art / "dev_pair_hand_scored.parquet")
    write_parquet(eval_scored, art / "eval_pair_hand_scored.parquet")
    joblib.dump(full_evidence, art / "evidence_model.joblib")
    (art / "evidence_features.json").write_text(json.dumps(evidence_features, indent=2), encoding="utf-8")

    top_k = tuple(cfg["features"].get("top_k", [1, 2, 3, 5, 10]))
    dev_pair = aggregate_pair_features(dev_scored, top_k=top_k)
    eval_pair = aggregate_pair_features(eval_scored, top_k=top_k)

    labels = pl.read_parquet(art / "development_labels_normalized.parquet")
    schema_dict = json.loads((art / "schema.json").read_text(encoding="utf-8"))
    pair_id_source = schema_dict["pair_id"]
    labels_for_join = labels.select([
        pl.col(pair_id_source).alias("pair_id") if pair_id_source in labels.columns else pl.col("pair_id"),
        "y_pair",
        "behavior",
    ])
    dev_pair = dev_pair.join(labels_for_join, on="pair_id", how="left")

    pair_features = model_feature_columns(dev_pair)
    X_pair = _to_pandas_features(dev_pair, pair_features)
    y_pair = dev_pair["y_pair"].to_numpy()
    pair_groups = _pair_groups(dev_pair)

    pair_oof, _ = fit_binary_oof(X_pair, y_pair, pair_groups, cfg["models"]["pair"], n_splits, seed + 100)
    pair_ap = float(average_precision_score(y_pair, pair_oof.oof))
    full_pair = fit_full_binary(X_pair, y_pair, cfg["models"]["pair"], seed + 100)

    X_eval_pair = _to_pandas_features(eval_pair, pair_features)
    eval_pair_raw = full_pair.predict_proba(X_eval_pair)[:, 1]
    eval_risk = rank_normalize(eval_pair_raw)

    joblib.dump(full_pair, art / "pair_model.joblib")
    (art / "pair_features.json").write_text(json.dumps(pair_features, indent=2), encoding="utf-8")

    disclosed = list(cfg["behavior"].get("disclosed_classes", sorted(DISCLOSED_BEHAVIORS)))
    behavior_oof: dict[str, np.ndarray] = {}
    behavior_eval: dict[str, np.ndarray] = {}
    behavior_models = {}
    for i, cls in enumerate(disclosed):
        y_cls = (dev_pair["behavior"].to_numpy() == cls).astype(int)
        oof_cls, _ = fit_binary_oof(
            X_pair,
            y_cls,
            pair_groups,
            cfg["models"]["behavior"],
            n_splits,
            seed + 200 + i * 13,
        )
        behavior_oof[cls] = oof_cls.oof
        model = fit_full_binary(X_pair, y_cls, cfg["models"]["behavior"], seed + 200 + i * 13)
        behavior_models[cls] = model
        behavior_eval[cls] = model.predict_proba(X_eval_pair)[:, 1]

    joblib.dump(behavior_models, art / "behavior_models.joblib")

    best = (-1.0, None, None)
    y_behavior = dev_pair["behavior"].to_numpy()
    for none_threshold in cfg["behavior"]["none_threshold_grid"]:
        for known_confidence in cfg["behavior"]["known_confidence_grid"]:
            pred = choose_behaviors(pair_oof.oof, behavior_oof, none_threshold, known_confidence)
            score = behavior_map(y_behavior, pair_oof.oof, pred)
            if score > best[0]:
                best = (score, float(none_threshold), float(known_confidence))
    behavior_score, none_threshold, known_confidence = best

    metrics = {
        "evidence_clean_fold_ap": evidence_oof_result.fold_ap,
        "pair_oof_ap": pair_ap,
        "behavior_oof_map": behavior_score,
        "none_threshold": none_threshold,
        "known_confidence": known_confidence,
    }
    (art / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    eval_pred = pd.DataFrame({
        "pair_id": eval_pair["pair_id"].to_list(),
        "risk_score": eval_risk,
    })
    for cls, arr in behavior_eval.items():
        eval_pred[f"prob_{cls}"] = arr
    eval_pred.to_parquet(art / "eval_pair_predictions.parquet", index=False)
    eval_pair.write_parquet(art / "eval_pair_features.parquet")
    return metrics


def predict_submission(cfg: dict, output: str | Path = "submission.csv") -> Path:
    paths = resolve_paths(cfg)
    art = paths["artifacts_dir"]
    metrics = json.loads((art / "metrics.json").read_text(encoding="utf-8"))
    pair_pred = pd.read_parquet(art / "eval_pair_predictions.parquet")
    eval_ph = pl.read_parquet(art / "eval_pair_hand_scored.parquet").select(["pair_id", "hand_id", "evidence_score"]).to_pandas()

    classes = cfg["behavior"].get("disclosed_classes", ["directed_transfer", "soft_play", "coordinated_isolation"])
    sample = pd.read_csv(paths["sample_submission"])
    aligned = sample[["pair_id"]].merge(pair_pred, on="pair_id", how="left", validate="one_to_one")
    if aligned["risk_score"].isna().any():
        raise ValueError("Some sample pair_ids have no prediction")

    class_probs_aligned = {c: aligned[f"prob_{c}"].to_numpy() for c in classes}
    behavior_aligned = choose_behaviors(
        aligned["risk_score"].to_numpy(),
        class_probs_aligned,
        metrics["none_threshold"],
        metrics["known_confidence"],
    )

    evidence = top_evidence(eval_ph, sample["pair_id"], k=5)
    sub = build_submission(sample, aligned["risk_score"].to_numpy(), behavior_aligned, evidence)
    output = Path(output)
    sub.to_csv(output, index=False)
    return output
