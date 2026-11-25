"""测试集评估：计算指标、保存混淆矩阵/分类报告/误分类样本等报告素材。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .engine import evaluate
from .metrics import (
    classification_report,
    confusion_matrix,
    per_class_accuracy,
    topk_accuracy,
    top_confusions,
)
from .utils import save_json
from .visualization import (
    collect_errors,
    plot_confidence_distribution,
    plot_confusion_matrix,
    plot_misclassified_grid,
    plot_per_class_accuracy,
    save_misclassified_csv,
)


def run_test_evaluation(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    raw_test_dataset: Any,
    device: torch.device,
    output_dir: str | Path,
    class_names: Sequence[str],
    criterion: Optional[nn.Module] = None,
    amp: bool = False,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """在测试集上评估并生成全部报告素材。

    输出文件：
        test_metrics.json              全部数值指标
        test_predictions.npz           逐样本 targets / preds / probs，用于配对检验
        classification_report.txt      每类 precision/recall/F1
        confusion_matrix.csv           混淆矩阵（行=真实，列=预测）
        confusion_matrix.png           混淆矩阵计数图
        confusion_matrix_normalized.png 按真实类别归一化的混淆矩阵
        per_class_accuracy.png         每类准确率
        confidence_distribution.png    正确/错误预测的置信度分布
        misclassified_samples.csv      全部误分类样本（按置信度降序）
        misclassified_samples.png      置信度最高的误分类样本网格
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        amp=amp,
        max_batches=max_batches,
        desc="Test",
    )
    targets = result["targets"]
    preds = result["preds"]
    probs = result["probs"]

    num_classes = len(class_names)
    cm = confusion_matrix(targets, preds, num_classes)
    class_accuracy, support = per_class_accuracy(cm)
    report, report_text = classification_report(targets, preds, class_names)
    errors = collect_errors(targets, preds, probs, class_names)
    confusions = top_confusions(cm, class_names, top_k=10)

    confidence = probs.max(axis=1)
    correct_mask = preds == targets

    metrics: Dict[str, Any] = {
        "num_samples": int(len(targets)),
        "loss": float(result["loss"]),
        "accuracy": float(report["accuracy"]),
        "error_rate": float(1.0 - report["accuracy"]),
        "top3_accuracy": topk_accuracy(probs, targets, k=3),
        "macro_precision": report["macro_precision"],
        "macro_recall": report["macro_recall"],
        "macro_f1": report["macro_f1"],
        "weighted_precision": report["weighted_precision"],
        "weighted_recall": report["weighted_recall"],
        "weighted_f1": report["weighted_f1"],
        "num_errors": len(errors),
        "mean_confidence_correct": (
            float(confidence[correct_mask].mean()) if correct_mask.any() else None
        ),
        "mean_confidence_incorrect": (
            float(confidence[~correct_mask].mean()) if (~correct_mask).any() else None
        ),
        "per_class": report["per_class"],
        "top_confusions": confusions,
        "predictions_file": "test_predictions.npz",
    }
    for index, name in enumerate(class_names):
        metrics["per_class"][name]["accuracy"] = float(class_accuracy[index])
        metrics["per_class"][name]["support"] = int(support[index])

    save_json(output_dir / "test_metrics.json", metrics)
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        targets=np.asarray(targets, dtype=np.int64),
        preds=np.asarray(preds, dtype=np.int64),
        probs=np.asarray(probs, dtype=np.float32),
        class_names=np.asarray(class_names),
    )
    np.savetxt(
        output_dir / "confusion_matrix.csv",
        cm,
        fmt="%d",
        delimiter=",",
        header=",".join(class_names),
        comments="",
    )

    report_lines = [
        "CIFAR-10 Test Set Classification Report",
        "=" * 57,
        f"Samples        : {len(targets)}",
        f"Loss           : {result['loss']:.4f}",
        f"Accuracy       : {report['accuracy']:.4f}",
        f"Top-3 accuracy : {metrics['top3_accuracy']:.4f}",
        f"Macro F1       : {report['macro_f1']:.4f}",
        f"Weighted F1    : {report['weighted_f1']:.4f}",
        f"Errors         : {len(errors)}",
        "",
        report_text,
        "",
        "Top confusions (true -> predicted):",
    ]
    for item in confusions:
        report_lines.append(
            f"  {item['true_label']:>10} -> {item['predicted_label']:<10} "
            f"{item['count']:>4} ({item['rate']:.2%} of true class)"
        )
    (output_dir / "classification_report.txt").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    plot_confusion_matrix(
        cm, class_names, output_dir / "confusion_matrix.png", normalize=False
    )
    plot_confusion_matrix(
        cm,
        class_names,
        output_dir / "confusion_matrix_normalized.png",
        normalize=True,
    )
    plot_per_class_accuracy(
        class_accuracy, class_names, output_dir / "per_class_accuracy.png"
    )
    plot_confidence_distribution(
        targets, preds, probs, output_dir / "confidence_distribution.png"
    )
    save_misclassified_csv(errors, output_dir / "misclassified_samples.csv")
    plot_misclassified_grid(
        raw_test_dataset, errors, output_dir / "misclassified_samples.png"
    )

    print(
        f"[Test] samples={metrics['num_samples']} loss={metrics['loss']:.4f} "
        f"acc={metrics['accuracy']:.4f} macro-F1={metrics['macro_f1']:.4f}"
    )
    print(
        f"[Test] Top-3 acc={metrics['top3_accuracy']:.4f} "
        f"errors={metrics['num_errors']} "
        f"(mean confidence: correct={metrics['mean_confidence_correct']}, "
        f"incorrect={metrics['mean_confidence_incorrect']})"
    )
    if confusions:
        print("[Test] Top confusions:")
        for item in confusions[:5]:
            print(
                f"       {item['true_label']:>10} -> {item['predicted_label']:<10} "
                f"{item['count']:>4} ({item['rate']:.2%})"
            )
    print(f"[Test] 评估结果已保存到：{output_dir.resolve()}")

    return metrics
