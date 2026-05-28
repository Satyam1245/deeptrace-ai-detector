"""
Evaluation metrics for deepfake detection.
Includes EER, ACER, APCER, NPCER, and other forensic metrics.
"""

import numpy as np
import math
import logging
from typing import Tuple, List, Dict, Optional

from sklearn.metrics import accuracy_score, roc_auc_score

logger = logging.getLogger(__name__)


def eval_state(probs: np.ndarray, labels: np.ndarray, thr: float) -> Tuple[int, int, int, int]:
    predict = probs >= thr
    TN = np.sum((labels == 0) & (predict == False))
    FN = np.sum((labels == 1) & (predict == False))
    FP = np.sum((labels == 0) & (predict == True))
    TP = np.sum((labels == 1) & (predict == True))
    return TN, FN, FP, TP


def calculate_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:

    TN, FN, FP, TP = eval_state(probs, labels, threshold)

    APCER = FP / (FP + TN) if (FP + TN) > 0 else 1.0
    NPCER = FN / (FN + TP) if (FN + TP) > 0 else 1.0
    ACER = (APCER + NPCER) / 2.0

    ACC = (TP + TN) / (TN + FN + FP + TP) if (TN + FN + FP + TP) > 0 else 0.0
    PRECISION = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    RECALL = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    F1 = (2 * PRECISION * RECALL) / (PRECISION + RECALL) if (PRECISION + RECALL) > 0 else 0.0
    SPECIFICITY = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    return {
        "accuracy": ACC,
        "apcer": APCER,
        "npcer": NPCER,
        "acer": ACER,
        "precision": PRECISION,
        "recall": RECALL,
        "f1_score": F1,
        "specificity": SPECIFICITY,
        "tp": int(TP),
        "tn": int(TN),
        "fp": int(FP),
        "fn": int(FN),
    }


def get_threshold(grid_density: int = 10000) -> List[float]:
    thresholds = [i / float(grid_density) for i in range(grid_density + 1)]
    thresholds.append(1.1)
    return thresholds


def get_EER_states(
    probs: np.ndarray,
    labels: np.ndarray,
    grid_density: int = 10000
) -> Tuple[float, float, List[float], List[float]]:

    thresholds = get_threshold(grid_density)
    min_dist = 1.0
    min_state = (1.0, 1.0, 0.5)
    FRR_list, FAR_list = [], []

    for thr in thresholds:
        TN, FN, FP, TP = eval_state(probs, labels, thr)

        FAR = FP / (FP + TN) if (FP + TN) > 0 else 1.0
        FRR = FN / (FN + TP) if (FN + TP) > 0 else 1.0

        FAR_list.append(FAR)
        FRR_list.append(FRR)

        dist = abs(FAR - FRR)
        if dist < min_dist:
            min_dist = dist
            min_state = (FAR, FRR, thr)

    EER = (min_state[0] + min_state[1]) / 2.0
    return EER, min_state[2], FRR_list, FAR_list


def get_HTER_at_thr(probs: np.ndarray, labels: np.ndarray, thr: float) -> float:
    TN, FN, FP, TP = eval_state(probs, labels, thr)

    FAR = FP / (FP + TN) if (FP + TN) > 0 else 1.0
    FRR = FN / (FN + TP) if (FN + TP) > 0 else 1.0

    return (FAR + FRR) / 2.0


def calculate_comprehensive_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    preds: Optional[np.ndarray] = None
) -> Dict[str, float]:

    if preds is None:
        preds = (probs >= 0.5).astype(int)

    metrics = calculate_metrics(probs, labels)

    EER, optimal_thr, _, _ = get_EER_states(probs, labels)
    metrics["eer"] = EER
    metrics["optimal_threshold"] = optimal_thr
    metrics["hter"] = get_HTER_at_thr(probs, labels, 0.5)

    try:
        metrics["auc_roc"] = roc_auc_score(labels, probs)
    except Exception:
        logger.warning("AUC-ROC could not be calculated")
        metrics["auc_roc"] = 0.0

    return metrics
