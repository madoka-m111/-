"""分类评估指标：混淆矩阵、每类准确率、precision/recall/F1、Top-k 准确率。"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def confusion_matrix(
    targets: Sequence[int],
    preds: Sequence[int],
    num_classes: int,
) -> np.ndarray:
    """计算混淆矩阵，行表示真实类别，列表示预测类别。"""
    targets_array = np.asarray(targets, dtype=np.int64)
    preds_array = np.asarray(preds, dtype=np.int64)
    if targets_array.shape != preds_array.shape:
        raise ValueError("targets 与 preds 的形状必须一致。")

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (targets_array, preds_array), 1)
    return cm


def per_class_accuracy(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """由混淆矩阵计算每类准确率与样本数。

    Returns:
        (class_accuracy, support)，长度均为 num_classes。
    """
    support = cm.sum(axis=1)
    accuracy = np.divide(
        cm.diagonal(),
        support,
        out=np.zeros(len(support), dtype=np.float64),
        where=support > 0,
    )
    return accuracy, support


def classification_report(
    targets: Sequence[int],
    preds: Sequence[int],
    class_names: Sequence[str],
) -> Tuple[Dict, str]:
    """计算每类 precision / recall / F1 以及 macro、weighted 平均值。

    Returns:
        (report_dict, report_text)
    """
    num_classes = len(class_names)
    cm = confusion_matrix(targets, preds, num_classes)
    total = int(cm.sum())
    accuracy = float(np.trace(cm) / total) if total > 0 else 0.0

    per_class: Dict[str, Dict[str, float]] = {}
    precisions: List[float] = []
    recalls: List[float] = []
    f1_scores: List[float] = []
    supports: List[int] = []

    for i, name in enumerate(class_names):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - tp)
        fn = float(cm[i, :].sum() - tp)
        support = int(cm[i, :].sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        if support > 0:
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            supports.append(support)

    def _weighted(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        weights = np.asarray(supports, dtype=np.float64)
        return float(np.average(values, weights=weights))

    report = {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "weighted_precision": _weighted(precisions),
        "weighted_recall": _weighted(recalls),
        "weighted_f1": _weighted(f1_scores),
        "per_class": per_class,
    }

    lines = [
        f"{'class':>12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>9}",
        "-" * 57,
    ]
    for name in class_names:
        item = per_class[name]
        lines.append(
            f"{name:>12} {item['precision']:>10.4f} {item['recall']:>10.4f} "
            f"{item['f1']:>10.4f} {item['support']:>9d}"
        )
    lines.extend(
        [
            "-" * 57,
            f"{'macro avg':>12} {report['macro_precision']:>10.4f} "
            f"{report['macro_recall']:>10.4f} {report['macro_f1']:>10.4f} "
            f"{total:>9d}",
            f"{'weighted avg':>12} {report['weighted_precision']:>10.4f} "
            f"{report['weighted_recall']:>10.4f} {report['weighted_f1']:>10.4f} "
            f"{total:>9d}",
            f"{'accuracy':>12} {'':>10} {'':>10} {accuracy:>10.4f} {total:>9d}",
        ]
    )
    return report, "\n".join(lines)


def topk_accuracy(
    probs: np.ndarray,
    targets: Sequence[int],
    k: int = 3,
) -> float:
    """计算 Top-k 准确率。"""
    probs = np.asarray(probs)
    targets_array = np.asarray(targets, dtype=np.int64)
    if probs.ndim != 2:
        raise ValueError("probs 必须是二维数组 [num_samples, num_classes]。")

    k = min(k, probs.shape[1])
    topk_indices = np.argsort(-probs, axis=1)[:, :k]
    matches = topk_indices == targets_array[:, None]
    return float(matches.any(axis=1).mean())


def top_confusions(
    cm: np.ndarray,
    class_names: Sequence[str],
    top_k: int = 10,
) -> List[Dict]:
    """提取最常见的“真实类别 -> 预测类别”混淆对（不含对角线）。"""
    pairs: List[Dict] = []
    row_sums = cm.sum(axis=1)

    for true_id in range(len(class_names)):
        for pred_id in range(len(class_names)):
            if true_id == pred_id:
                continue
            count = int(cm[true_id, pred_id])
            if count <= 0:
                continue
            support = int(row_sums[true_id])
            pairs.append(
                {
                    "true_label": class_names[true_id],
                    "predicted_label": class_names[pred_id],
                    "count": count,
                    "rate": float(count / support) if support > 0 else 0.0,
                }
            )

    pairs.sort(key=lambda item: (-item["count"], item["true_label"]))
    return pairs[:top_k]
