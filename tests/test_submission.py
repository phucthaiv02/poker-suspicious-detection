import numpy as np
import pandas as pd
from poker_collusion.submission import choose_behaviors, top_evidence


def test_choose_behaviors_rejects_unknown():
    risk = np.array([0.9, 0.9, 0.01])
    probs = {
        "directed_transfer": np.array([0.8, 0.2, 0.9]),
        "soft_play": np.array([0.1, 0.25, 0.05]),
        "coordinated_isolation": np.array([0.1, 0.2, 0.05]),
    }
    out = choose_behaviors(risk, probs, none_threshold=0.1, known_confidence=0.5)
    assert out.tolist() == ["directed_transfer", "other_coordination", "none"]


def test_top_evidence_unique_ranked():
    ph = pd.DataFrame({
        "pair_id": [1, 1, 1, 2],
        "hand_id": [12, 11, 11, 20],
        "evidence_score": [0.8, 0.9, 0.7, 0.5],
    })
    out = top_evidence(ph, pd.Series([1, 2]))
    assert out.loc[0, "evidence_hand_1"] == 11
    assert out.loc[0, "evidence_hand_2"] == 12
    assert out.loc[1, "evidence_hand_2"] == "NO_EVIDENCE"
