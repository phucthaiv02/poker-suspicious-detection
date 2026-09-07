import pandas as pd
from poker_collusion.metrics import evidence_map5


def test_evidence_map5_perfect():
    truth = pd.DataFrame({"pair_id": [1, 1, 2], "hand_id": [10, 11, 20]})
    pred = pd.DataFrame({
        "pair_id": [1, 2],
        "evidence_hand_1": [10, 20],
        "evidence_hand_2": [11, "NO_EVIDENCE"],
        "evidence_hand_3": ["NO_EVIDENCE", "NO_EVIDENCE"],
        "evidence_hand_4": ["NO_EVIDENCE", "NO_EVIDENCE"],
        "evidence_hand_5": ["NO_EVIDENCE", "NO_EVIDENCE"],
    })
    assert evidence_map5(truth, pred) == 1.0
