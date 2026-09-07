from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import polars as pl

TARGET_BEHAVIORS = {"directed_transfer", "soft_play", "coordinated_isolation", "other_coordination"}
DISCLOSED_BEHAVIORS = {"directed_transfer", "soft_play", "coordinated_isolation"}


def _first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    cols = set(columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def _detect_pair_player_columns(columns: list[str], pair_id_col: str) -> tuple[str, str]:
    preferred = [
        ("player_id_1", "player_id_2"),
        ("player1_id", "player2_id"),
        ("player_a_id", "player_b_id"),
        ("player_a", "player_b"),
        ("player_1", "player_2"),
        ("p1", "p2"),
    ]
    cols = set(columns)
    for a, b in preferred:
        if a in cols and b in cols:
            return a, b
    candidates = [
        c for c in columns
        if c != pair_id_col and "player" in c.lower() and ("id" in c.lower() or c.lower().startswith("player"))
    ]
    if len(candidates) == 2:
        return candidates[0], candidates[1]
    raise ValueError(
        "Could not detect the two player columns in a pair file. "
        f"Columns={columns}. Add schema_overrides.pair_player_1 / pair_player_2."
    )


def _read_columns(path: Path) -> list[str]:
    if path.suffix.lower() == ".parquet":
        return pl.scan_parquet(path).collect_schema().names()
    return pl.read_csv(path, n_rows=3).columns


@dataclass
class Schema:
    hand_id: str
    player_id: str
    table_id: str | None
    phase: str | None
    action_type: str | None
    action_order: str | None
    pair_id: str
    pair_player_1: str
    pair_player_2: str
    label: str | None
    behavior: str | None
    evidence_hand: str

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


def detect_schema(paths: dict[str, Path], overrides: dict | None = None) -> Schema:
    overrides = overrides or {}
    hands_cols = _read_columns(paths["hands"])
    seats_cols = _read_columns(paths["seats"])
    actions_cols = _read_columns(paths["actions"])
    eval_cols = _read_columns(paths["evaluation_pairs"])
    label_cols = _read_columns(paths["development_labels"])
    evidence_cols = _read_columns(paths["development_evidence"])

    hand_id = overrides.get("hand_id") or _first_existing(
        list(set(hands_cols) & set(seats_cols) & set(actions_cols)),
        ["hand_id", "hand", "hand_no", "hand_number"],
    )
    if not hand_id:
        raise ValueError("Could not detect hand_id shared by hands/seats/actions.")
    player_id = overrides.get("player_id") or _first_existing(
        list(set(seats_cols) & set(actions_cols)), ["player_id", "player", "account_id"]
    )
    if not player_id:
        raise ValueError("Could not detect player_id shared by seats/actions.")
    pair_id = overrides.get("pair_id") or _first_existing(eval_cols, ["pair_id", "pair"])
    if not pair_id:
        raise ValueError("Could not detect pair_id in evaluation_pairs.csv")

    if overrides.get("pair_player_1") and overrides.get("pair_player_2"):
        p1, p2 = overrides["pair_player_1"], overrides["pair_player_2"]
    else:
        p1, p2 = _detect_pair_player_columns(eval_cols, pair_id)

    table_id = overrides.get("table_id") or _first_existing(hands_cols, ["table_id", "table"])
    phase = overrides.get("phase") or _first_existing(hands_cols, ["phase", "split", "period"])
    action_type = overrides.get("action_type") or _first_existing(actions_cols, ["action_type", "action", "action_name", "action_kind"])
    action_order = overrides.get("action_order") or _first_existing(actions_cols, ["action_order", "action_index", "sequence", "action_no", "action_number"])
    behavior = overrides.get("behavior") or _first_existing(label_cols, ["behavior", "behavior_type", "target_behavior", "class"])
    label = overrides.get("label") or _first_existing(label_cols, ["label", "target", "is_target", "is_positive", "coordinated"])
    evidence_hand = overrides.get("evidence_hand") or _first_existing(evidence_cols, ["hand_id", "evidence_hand", "evidence_hand_id"])
    if not evidence_hand:
        raise ValueError("Could not detect evidence hand ID column.")

    return Schema(hand_id, player_id, table_id, phase, action_type, action_order, pair_id, p1, p2, label, behavior, evidence_hand)


def numeric_columns(path: Path, exclude: set[str]) -> list[str]:
    schema = pl.scan_parquet(path).collect_schema()
    numeric_dtypes = {pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64}
    return [name for name, dtype in schema.items() if name not in exclude and dtype in numeric_dtypes]
