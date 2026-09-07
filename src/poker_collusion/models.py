from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score


@dataclass
class OOFResult:
    oof: np.ndarray
    models: list[lgb.LGBMClassifier]
    fold_ap: list[float]


def make_lgbm(params: dict, seed: int, objective: str = "binary") -> lgb.LGBMClassifier:
    kwargs = dict(params)
    kwargs.setdefault("random_state", seed)
    kwargs.setdefault("n_jobs", -1)
    kwargs.setdefault("verbosity", -1)
    if objective == "binary":
        kwargs.setdefault("objective", "binary")
        kwargs.setdefault("class_weight", "balanced")
    return lgb.LGBMClassifier(**kwargs)


def fit_binary_oof(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    params: dict,
    n_splits: int,
    seed: int,
    predict_X: pd.DataFrame | None = None,
    predict_groups: np.ndarray | None = None,
) -> tuple[OOFResult, np.ndarray | None]:
    """Group OOF trainer.

    If predict_X/predict_groups are supplied, each fold model predicts only rows whose
    group belongs to that fold's validation groups. This is useful for scoring all
    hands of a held-out pair/pool while training evidence models on clean rows only.
    """
    splitter = GroupKFold(n_splits=n_splits)
    oof = np.full(len(X), np.nan, dtype=float)
    models: list[lgb.LGBMClassifier] = []
    fold_ap: list[float] = []
    pred_extra = np.full(len(predict_X), np.nan, dtype=float) if predict_X is not None else None

    for fold, (tr, va) in enumerate(splitter.split(X, y, groups)):
        model = make_lgbm(params, seed + fold)
        model.fit(X.iloc[tr], y[tr])
        p = model.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        if len(np.unique(y[va])) > 1:
            fold_ap.append(float(average_precision_score(y[va], p)))
        else:
            fold_ap.append(float("nan"))

        if predict_X is not None and predict_groups is not None:
            val_groups = set(groups[va].tolist())
            mask = np.fromiter((g in val_groups for g in predict_groups), dtype=bool, count=len(predict_groups))
            if mask.any():
                pred_extra[mask] = model.predict_proba(predict_X.loc[mask])[:, 1]
        models.append(model)

    return OOFResult(oof=oof, models=models, fold_ap=fold_ap), pred_extra


def fit_full_binary(X: pd.DataFrame, y: np.ndarray, params: dict, seed: int) -> lgb.LGBMClassifier:
    model = make_lgbm(params, seed)
    model.fit(X, y)
    return model


def predict_ensemble(models: Iterable[lgb.LGBMClassifier], X: pd.DataFrame) -> np.ndarray:
    preds = [m.predict_proba(X)[:, 1] for m in models]
    return np.mean(preds, axis=0)
