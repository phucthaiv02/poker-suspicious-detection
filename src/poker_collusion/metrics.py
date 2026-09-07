from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

DISCLOSED = ["directed_transfer", "soft_play", "coordinated_isolation"]


def pair_ap(y_true: Iterable[int], risk_score: Iterable[float]) -> float:
    return float(average_precision_score(np.asarray(y_true), np.asarray(risk_score)))


def behavior_map(y_behavior: Iterable[str], risk_score: Iterable[float], predicted_behavior: Iterable[str]) -> float:
    y_behavior = np.asarray(list(y_behavior), dtype=object)
    risk = np.asarray(list(risk_score), dtype=float)
    pred = np.asarray(list(predicted_behavior), dtype=object)
    aps = []
    for cls in DISCLOSED:
        y = (y_behavior == cls).astype(int)
        score = np.where(pred == cls, risk, 0.0)
        aps.append(float(average_precision_score(y, score)) if y.sum() > 0 else 0.0)
    return float(np.mean(aps))


def evidence_map5(
    true_evidence: pd.DataFrame,
    predictions: pd.DataFrame,
    pair_col: str = "pair_id",
    hand_col: str = "hand_id",
) -> float:
    """Development helper matching the textual MAP@5 definition.

    true_evidence must contain one row per relevant pair/hand. predictions should have
    pair_id and evidence_hand_1..5. Only pairs present in true_evidence are scored.
    """
    truth = defaultdict(set)
    for pair, hand in true_evidence[[pair_col, hand_col]].itertuples(index=False):
        truth[pair].add(hand)

    pred_map = predictions.set_index(pair_col)
    scores = []
    for pair, relevant in truth.items():
        if pair not in pred_map.index:
            scores.append(0.0)
            continue
        row = pred_map.loc[pair]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        ranked = [row.get(f"evidence_hand_{i}") for i in range(1, 6)]
        ranked = [h for h in ranked if h is not None and h != "NO_EVIDENCE"]
        hits = 0
        ap = 0.0
        for rank, hand in enumerate(ranked, start=1):
            if hand in relevant:
                hits += 1
                ap += hits / rank
        denom = min(len(relevant), 5)
        scores.append(ap / denom if denom else 0.0)
    return float(np.mean(scores)) if scores else 0.0
