"""可视化工具：训练曲线、混淆矩阵、每类准确率、置信度分布与误分类样本。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")  # 无界面服务器也能保存图片
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402


def _save_figure(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(
    history: Sequence[Dict[str, float]],
    save_path: str | Path,
) -> None:
    """绘制训练/验证 loss 与 accuracy 曲线，并标出最佳验证 epoch。"""
    if not history:
        return

    epochs = [int(item["epoch"]) for item in history]
    train_loss = [float(item["train_loss"]) for item in history]
    val_loss = [float(item["val_loss"]) for item in history]
    train_acc = [float(item["train_acc"]) * 100 for item in history]
    val_acc = [float(item["val_acc"]) * 100 for item in history]
    best_record = max(history, key=lambda item: float(item["val_acc"]))
    best_epoch = int(best_record["epoch"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, train_loss, label="Train", linewidth=1.5)
    axes[0].plot(epochs, val_loss, label="Validation", linewidth=1.5)
    axes[0].set_title("Loss curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")

    axes[1].plot(epochs, train_acc, label="Train", linewidth=1.5)
    axes[1].plot(epochs, val_acc, label="Validation", linewidth=1.5)
    axes[1].set_title("Accuracy curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")

    for ax in axes:
        ax.axvline(
            best_epoch,
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"Best epoch ({best_epoch})",
        )
        ax.set_xlim(0.5, max(epochs) + 0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    _save_figure(fig, save_path)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    save_path: str | Path,
    normalize: bool = False,
) -> None:
    """绘制混淆矩阵；normalize=True 时按真实类别归一化。"""
    cm = np.asarray(cm, dtype=np.float64)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        data = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
        title = "Normalized confusion matrix"
        value_format = ".2f"
        colorbar_label = "Proportion"
    else:
        data = cm
        title = "Confusion matrix"
        value_format = ".0f"
        colorbar_label = "Count"

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    vmax = data.max() if data.max() > 0 else 1.0
    image = ax.imshow(data, interpolation="nearest", cmap="Blues", vmin=0, vmax=vmax)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    threshold = vmax / 2.0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(
                j,
                i,
                format(data[i, j], value_format),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if data[i, j] > threshold else "black",
            )

    fig.tight_layout()
    _save_figure(fig, save_path)


def plot_per_class_accuracy(
    class_accuracy: Sequence[float],
    class_names: Sequence[str],
    save_path: str | Path,
) -> None:
    """绘制测试集每类准确率柱状图。"""
    values = np.asarray(class_accuracy, dtype=np.float64) * 100
    fig, ax = plt.subplots(figsize=(10, 4.8))
    positions = np.arange(len(class_names))
    bars = ax.bar(positions, values, color="steelblue", edgecolor="white")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 1.5, 100.0),
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.axhline(10.0, color="crimson", linestyle="--", linewidth=1, label="Chance (10%)")
    ax.set_xticks(positions)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Per-class test accuracy")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save_figure(fig, save_path)


def plot_confidence_distribution(
    targets: Sequence[int],
    preds: Sequence[int],
    probs: np.ndarray,
    save_path: str | Path,
) -> None:
    """绘制正确/错误预测的最大 softmax 概率分布，用于分析置信度校准。"""
    targets = np.asarray(targets)
    preds = np.asarray(preds)
    confidence = np.asarray(probs).max(axis=1)
    correct_mask = preds == targets

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bins = np.linspace(0.0, 1.0, 21)
    if correct_mask.any():
        ax.hist(
            confidence[correct_mask],
            bins=bins,
            density=True,
            alpha=0.6,
            label=f"Correct (n={int(correct_mask.sum())})",
        )
    if (~correct_mask).any():
        ax.hist(
            confidence[~correct_mask],
            bins=bins,
            density=True,
            alpha=0.6,
            label=f"Incorrect (n={int((~correct_mask).sum())})",
        )

    ax.set_xlabel("Maximum softmax probability")
    ax.set_ylabel("Density")
    ax.set_title("Prediction confidence distribution")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, save_path)


def collect_errors(
    targets: Sequence[int],
    preds: Sequence[int],
    probs: np.ndarray,
    class_names: Sequence[str],
) -> List[Dict[str, Any]]:
    """收集所有误分类样本，并按预测置信度从高到低排序。"""
    targets = np.asarray(targets)
    preds = np.asarray(preds)
    probs = np.asarray(probs)

    records: List[Dict[str, Any]] = []
    for index in np.flatnonzero(targets != preds):
        i = int(index)
        true_id = int(targets[i])
        pred_id = int(preds[i])
        records.append(
            {
                "image_index": i,
                "true_label": str(class_names[true_id]),
                "predicted_label": str(class_names[pred_id]),
                "pred_confidence": float(probs[i, pred_id]),
                "true_class_probability": float(probs[i, true_id]),
            }
        )

    records.sort(key=lambda item: (-item["pred_confidence"], item["image_index"]))
    return records


def save_misclassified_csv(
    records: Sequence[Dict[str, Any]],
    save_path: str | Path,
) -> None:
    """将误分类样本信息保存为 CSV（无错误时也写出表头）。"""
    fieldnames = [
        "image_index",
        "true_label",
        "predicted_label",
        "pred_confidence",
        "true_class_probability",
    ]
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def plot_misclassified_grid(
    raw_dataset: Any,
    records: Sequence[Dict[str, Any]],
    save_path: str | Path,
    max_items: int = 24,
    cols: int = 6,
) -> None:
    """可视化置信度最高的若干误分类样本。

    Args:
        raw_dataset: 返回 (PIL Image, label) 的原始测试集。
        records: collect_errors 的输出（已按置信度降序排列）。
        max_items: 最多展示的样本数。
        cols: 每行图片数。
    """
    if not records:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(
            0.5,
            0.5,
            "No misclassified samples",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.axis("off")
        _save_figure(fig, save_path)
        return

    items = list(records[:max_items])
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(2.6 * cols, 3.2 * rows),
        gridspec_kw={"hspace": 0.35, "wspace": 0.08},
    )
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.axis("off")

    for ax, record in zip(axes, items):
        sample = raw_dataset[int(record["image_index"])]
        image = sample[0] if isinstance(sample, (tuple, list)) else sample
        ax.imshow(image)
        ax.set_title(
            f"T: {record['true_label']}\n"
            f"P: {record['predicted_label']} ({record['pred_confidence']:.2f})",
            fontsize=8,
            color="darkred",
        )
        ax.axis("off")

    fig.suptitle("Most confident misclassified test images", fontsize=13)
    fig.subplots_adjust(top=0.93, bottom=0.01, left=0.01, right=0.99)
    _save_figure(fig, save_path)
