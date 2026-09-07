from __future__ import annotations

import numpy as np
import pandas as pd

VALID_BEHAVIORS = {"none", "directed_transfer", "soft_play", "coordinated_isolation", "other_coordination"}


def choose_behaviors(
    risk: np.ndarray,
    class_probs: dict[str, np.ndarray],
    none_threshold: float,
    known_confidence: float,
) -> np.ndarray:
    classes = list(class_probs)
    matrix = np.column_stack([class_probs[c] for c in classes])
    best_idx = matrix.argmax(axis=1)
    best_prob = matrix[np.arange(len(risk)), best_idx]
    best_class = np.asarray(classes, dtype=object)[best_idx]

    pred = best_class.astype(object)
    pred[risk < none_threshold] = "none"
    reject = (risk >= none_threshold) & (best_prob < known_confidence)
    pred[reject] = "other_coordination"
    return pred


def top_evidence(pair_hand_scores: pd.DataFrame, pair_ids: pd.Series, k: int = 5) -> pd.DataFrame:
    required = {"pair_id", "hand_id", "evidence_score"}
    missing = required - set(pair_hand_scores.columns)
    if missing:
        raise ValueError(f"Missing evidence columns: {missing}")

    ranked = (
        pair_hand_scores.sort_values(["pair_id", "evidence_score", "hand_id"], ascending=[True, False, True])
        .drop_duplicates(["pair_id", "hand_id"])
        .groupby("pair_id", sort=False)
        .head(k)
    )
    rows = []
    grouped = {pid: g["hand_id"].tolist() for pid, g in ranked.groupby("pair_id", sort=False)}
    for pid in pair_ids:
        hands = grouped.get(pid, [])[:k]
        hands = hands + ["NO_EVIDENCE"] * (k - len(hands))
        row = {"pair_id": pid}
        row.update({f"evidence_hand_{i+1}": hands[i] for i in range(k)})
        rows.append(row)
    return pd.DataFrame(rows)


def build_submission(
    sample: pd.DataFrame,
    risk_score: np.ndarray,
    predicted_behavior: np.ndarray,
    evidence: pd.DataFrame,
) -> pd.DataFrame:
    out = sample.copy()
    out["risk_score"] = np.clip(risk_score, 0.0, 1.0)
    out["predicted_behavior"] = predicted_behavior
    out = out.drop(columns=[c for c in out.columns if c.startswith("evidence_hand_")], errors="ignore")
    out = out.merge(evidence, on="pair_id", how="left")
    for i in range(1, 6):
        c = f"evidence_hand_{i}"
        out[c] = out[c].fillna("NO_EVIDENCE")
    if not set(out["predicted_behavior"]).issubset(VALID_BEHAVIORS):
        bad = sorted(set(out["predicted_behavior"]) - VALID_BEHAVIORS)
        raise ValueError(f"Invalid behaviors: {bad}")
    if out.isna().any().any():
        raise ValueError("Submission contains missing values")
    ev_cols = [f"evidence_hand_{i}" for i in range(1, 6)]
    duplicated = out[ev_cols].apply(
        lambda r: len([x for x in r if x != "NO_EVIDENCE"]) != len(set(x for x in r if x != "NO_EVIDENCE")), axis=1
    )
    if duplicated.any():
        raise ValueError("Repeated evidence hand within a pair")
    return out
