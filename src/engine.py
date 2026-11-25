"""训练循环、验证评估与 checkpoint 管理。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .utils import AverageMeter, load_json, save_json


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    amp: bool = False,
    scaler: Optional[Any] = None,
    grad_clip: float = 5.0,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """训练一个 epoch，返回平均 loss 与准确率。"""
    model.train()
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    amp_enabled = amp and device.type == "cuda"

    progress = tqdm(
        loader,
        desc=f"Train {epoch:03d}",
        leave=False,
        dynamic_ncols=True,
    )
    for batch_idx, (inputs, targets) in enumerate(progress):
        if max_batches is not None and batch_idx >= max_batches:
            break

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled
        ):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_size = targets.size(0)
        loss_meter.update(loss.item(), batch_size)
        correct += (outputs.argmax(dim=1) == targets).sum().item()
        total += batch_size

        progress.set_postfix(
            loss=f"{loss_meter.avg:.4f}",
            acc=f"{correct / max(total, 1):.4f}",
        )

    if total == 0:
        raise RuntimeError("训练 DataLoader 为空，请检查数据集与 --max-batches 设置。")

    return {"loss": loss_meter.avg, "acc": correct / total}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: Optional[nn.Module],
    device: torch.device,
    amp: bool = False,
    max_batches: Optional[int] = None,
    desc: str = "Eval",
) -> Dict[str, Any]:
    """在验证集/测试集上评估，返回 loss、准确率、标签、预测与概率。"""
    model.eval()
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    amp_enabled = amp and device.type == "cuda"

    all_targets: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []

    progress = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for batch_idx, (inputs, targets) in enumerate(progress):
        if max_batches is not None and batch_idx >= max_batches:
            break

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled
        ):
            logits = model(inputs)
            loss = criterion(logits, targets) if criterion is not None else None

        probs = torch.softmax(logits.float(), dim=1)
        preds = probs.argmax(dim=1)

        batch_size = targets.size(0)
        if loss is not None:
            loss_meter.update(loss.item(), batch_size)
        correct += (preds == targets).sum().item()
        total += batch_size

        all_targets.append(targets.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

        progress.set_postfix(
            loss=f"{loss_meter.avg:.4f}",
            acc=f"{correct / max(total, 1):.4f}",
        )

    if total == 0:
        raise RuntimeError("评估 DataLoader 为空，请检查数据集与 --max-batches 设置。")

    return {
        "loss": loss_meter.avg if loss_meter.count > 0 else float("nan"),
        "acc": correct / total,
        "targets": np.concatenate(all_targets),
        "preds": np.concatenate(all_preds),
        "probs": np.concatenate(all_probs),
    }


def _write_history_csv(history: List[Dict[str, float]], path: Path) -> None:
    """将训练历史写入 CSV，便于在 Excel / pandas 中分析。"""
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def fit(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    device: torch.device,
    epochs: int,
    output_dir: str | Path,
    amp: bool = False,
    grad_clip: float = 5.0,
    patience: int = 0,
    max_batches: Optional[int] = None,
    start_epoch: int = 0,
    best_val_acc: float = -1.0,
    extra_checkpoint: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, float]], float, int]:
    """完整训练流程：逐 epoch 训练、验证、保存最优模型、可选早停。

    Args:
        start_epoch: 从 0 开始计数；断点续训时等于已完成的 epoch 数。
        best_val_acc: 历史最佳验证准确率，用于断点续训时判断是否刷新最优。
        patience: 连续多少个 epoch 验证准确率不提升后早停；0 表示关闭。
        max_batches: 每个 epoch 最多训练的 batch 数，仅用于快速调试。

    Returns:
        (history, best_val_acc, best_epoch)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    amp_enabled = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None

    history_path = output_dir / "history.json"
    if start_epoch > 0 and history_path.exists():
        history = load_json(history_path)
        # 断点续训时只保留已完成的 epoch，避免重复记录
        history = [item for item in history if int(item.get("epoch", 0)) <= start_epoch]
    else:
        history = []

    best_epoch = -1
    if history:
        best_record = max(history, key=lambda item: float(item.get("val_acc", 0.0)))
        best_epoch = int(best_record["epoch"])

    epochs_no_improve = 0

    for epoch in range(start_epoch, epochs):
        current_lr = float(optimizer.param_groups[0]["lr"])

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch + 1,
            amp=amp,
            scaler=scaler,
            grad_clip=grad_clip,
            max_batches=max_batches,
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp=amp,
            max_batches=max_batches,
            desc="Val",
        )

        if scheduler is not None:
            scheduler.step()

        record = {
            "epoch": epoch + 1,
            "lr": current_lr,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
        }
        history.append(record)
        save_json(history_path, history)
        _write_history_csv(history, output_dir / "history.csv")

        is_best = val_metrics["acc"] > best_val_acc
        if is_best:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        payload: Dict[str, Any] = dict(extra_checkpoint or {})
        payload.update(
            {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler else None,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "history": history,
            }
        )
        torch.save(payload, output_dir / "last.pt")
        if is_best:
            torch.save(payload, output_dir / "best.pt")

        marker = " *" if is_best else ""
        print(
            f"Epoch {epoch + 1:03d}/{epochs:03d} | lr {current_lr:.5f} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f}{marker}"
        )

        if patience > 0 and epochs_no_improve >= patience:
            print(
                f"Early stopping: 验证准确率连续 {patience} 个 epoch 未提升，"
                f"停止训练（最佳 epoch = {best_epoch}）。"
            )
            break

    return history, best_val_acc, best_epoch
