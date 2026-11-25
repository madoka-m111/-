"""CIFAR-10 图像分类训练入口。

示例：
    python train.py --download --output-dir outputs/final100
    python train.py --epochs 100 --batch-size 128 --lr 0.1 --output-dir outputs/final100
    python train.py --scheduler none --epochs 100 --output-dir outputs/no_sched100
    python train.py --use-se --se-stages 1,2 --se-reduction 8 --se-init-bias 2.0
    python train.py --epochs 30 --skip-test --output-dir outputs/screen/base
    python train.py --no-augment --epochs 30 --output-dir outputs/no_aug
    python train.py --resume outputs/final100/last.pt --output-dir outputs/final100
"""

from __future__ import annotations

import argparse

import torch.nn as nn
import torch.optim as optim

from src.data import build_dataloaders
from src.engine import fit
from src.evaluation import run_test_evaluation
from src.model import build_model, parse_se_stages
from src.utils import (
    count_parameters,
    ensure_dir,
    load_checkpoint,
    resolve_device,
    save_json,
    set_seed,
)
from src.visualization import plot_training_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在 CIFAR-10 上训练自定义 CNN，并在测试集上评估。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 数据与输出
    parser.add_argument("--data-root", type=str, default="data", help="数据集根目录")
    parser.add_argument("--output-dir", type=str, default="outputs", help="结果输出目录")
    parser.add_argument(
        "--download", action="store_true", help="本地无数据时自动下载 CIFAR-10"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="DataLoader 工作进程数（Windows 可设 0）"
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="关闭训练集的 RandomCrop 与 RandomHorizontalFlip，仅用于数据增强对比",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--deterministic", action="store_true", help="开启 cuDNN 确定性模式（速度较慢）"
    )

    # 模型与训练
    parser.add_argument(
        "--model", type=str, default="cifar_cnn", choices=["cifar_cnn"], help="模型名称"
    )
    parser.add_argument(
        "--use-se",
        action="store_true",
        help="在 CIFARCNN 中按 --se-stages 选择 stage 加入 SE 通道注意力（默认关闭）",
    )
    parser.add_argument(
        "--se-reduction",
        type=int,
        default=16,
        help="SE 模块的通道压缩比，需 >= 1；仅 --use-se 生效",
    )
    parser.add_argument(
        "--se-stages",
        type=str,
        default="1,2,3",
        help=(
            "在哪些 stage 插入 SE，逗号分隔（1-based），例如 1 或 1,2；"
            "默认 1,2,3，保持旧行为"
        ),
    )
    parser.add_argument(
        "--se-init-bias",
        type=float,
        default=0.0,
        help=(
            "SE 最后一层 bias 初值；0 表示初始门控 0.5（旧行为），"
            "2.0 表示约 0.88，可用于近恒等初始化"
        ),
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="训练结束后跳过测试集评估，仅用于筛选/消融阶段，避免反复使用测试集",
    )
    parser.add_argument("--epochs", type=int, default=100, help="训练总 epoch 数")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="初始学习率（SGD）")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="权重衰减")
    parser.add_argument(
        "--label-smoothing", type=float, default=0.0, help="交叉熵 label smoothing 系数"
    )
    parser.add_argument("--grad-clip", type=float, default=5.0, help="梯度裁剪阈值，0 表示关闭")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="连续多少个 epoch 验证准确率不提升后早停，0 表示关闭",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "step", "none"],
        help="学习率调度器：cosine / step / none（默认 cosine，保持原有行为）",
    )
    parser.add_argument(
        "--scheduler-step-size",
        type=int,
        default=30,
        help="StepLR 的衰减间隔，仅 --scheduler step 生效",
    )
    parser.add_argument(
        "--scheduler-gamma",
        type=float,
        default=0.1,
        help="StepLR 的学习率衰减系数，仅 --scheduler step 生效",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="每个 epoch 最多训练的 batch 数，仅用于快速调试，0 表示不限制",
    )

    # 运行时
    parser.add_argument("--device", type=str, default="auto", help="auto / cpu / cuda / cuda:1")
    parser.add_argument("--amp", action="store_true", help="使用混合精度训练（仅 CUDA）")
    parser.add_argument("--resume", type=str, default="", help="从 checkpoint 断点续训")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    set_seed(args.seed, deterministic=args.deterministic)
    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)

    # 在训练开始前完成 stage 选择校验，避免跑完数据加载后才报错。
    try:
        se_stages = parse_se_stages(args.se_stages)
    except ValueError as exc:
        raise SystemExit(f"参数错误：{exc}") from exc

    if args.amp and device.type != "cuda":
        print("[Warn] --amp 仅在 CUDA 设备上生效，当前已自动关闭。")
        args.amp = False

    save_json(output_dir / "args.json", vars(args))
    print(f"Device: {device} | Output: {output_dir.resolve()}")

    data = build_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
        download=args.download,
        use_augment=not args.no_augment,
    )
    print(f"Train augmentation: {'off' if args.no_augment else 'on'}")
    print(
        f"Train: {len(data.train.dataset)} samples | "
        f"Val: {len(data.val.dataset)} samples | "
        f"Test: {len(data.test.dataset)} samples"
    )

    save_json(
        output_dir / "split_indices.json",
        {
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "num_train": len(data.train_indices),
            "num_val": len(data.val_indices),
            "train_indices": data.train_indices,
            "val_indices": data.val_indices,
        },
    )

    model = build_model(
        args.model,
        num_classes=len(data.class_names),
        use_se=args.use_se,
        se_reduction=args.se_reduction,
        se_stages=se_stages,
        se_init_bias=args.se_init_bias,
    ).to(device)
    num_parameters = count_parameters(model)
    print(
        f"Model: {args.model} | SE: {args.use_se} | "
        f"SE stages: {se_stages if args.use_se else 'n/a'} | "
        f"SE init bias: {args.se_init_bias} | "
        f"Trainable parameters: {num_parameters:,}"
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )
    elif args.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.scheduler_step_size,
            gamma=args.scheduler_gamma,
        )
    else:
        scheduler = None

    start_epoch = 0
    best_val_acc = -1.0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, map_location=device)
        saved_args = checkpoint.get("args") or {}
        for key in (
            "seed",
            "val_ratio",
            "model",
            "epochs",
            "scheduler",
            "no_augment",
            "use_se",
            "se_reduction",
            "se_stages",
            "se_init_bias",
        ):
            if key not in saved_args:
                continue
            saved_value = saved_args[key]
            current_value = getattr(args, key)
            if key == "se_stages":
                try:
                    saved_value = parse_se_stages(saved_value)
                    current_value = parse_se_stages(current_value)
                except ValueError as exc:
                    raise SystemExit(
                        f"checkpoint 中的 se_stages 无效：{exc}"
                    ) from exc
            if saved_value != current_value:
                raise SystemExit(
                    f"断点续训参数不一致：{key} "
                    f"checkpoint={saved_value} 当前={current_value}。"
                    "请保持数据划分与训练配置一致。"
                )

        model.load_state_dict(checkpoint["model_state"])
        if checkpoint.get("optimizer_state") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler is not None and checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_val_acc = float(checkpoint.get("best_val_acc", -1.0))

        # 若输出目录已更换，从 checkpoint 恢复训练历史
        history_path = output_dir / "history.json"
        if not history_path.exists() and checkpoint.get("history"):
            save_json(history_path, checkpoint["history"])

        print(
            f"Resume from {args.resume} | completed epoch={start_epoch} | "
            f"best_val_acc={best_val_acc:.4f}"
        )

    if start_epoch >= args.epochs:
        raise SystemExit(
            f"checkpoint 已完成 {start_epoch} 个 epoch，当前 --epochs={args.epochs}。"
            "请保持与原实验相同的 --epochs，或从头开始一次新的训练。"
        )

    max_batches = args.max_batches if args.max_batches > 0 else None
    history, best_val_acc, best_epoch = fit(
        model=model,
        train_loader=data.train,
        val_loader=data.val,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        output_dir=output_dir,
        amp=args.amp,
        grad_clip=args.grad_clip,
        patience=args.patience,
        max_batches=max_batches,
        start_epoch=start_epoch,
        best_val_acc=best_val_acc,
        extra_checkpoint={
            "args": vars(args),
            "model_name": args.model,
            "model_kwargs": {
                "use_se": args.use_se,
                "se_reduction": args.se_reduction,
                "se_stages": list(se_stages),
                "se_init_bias": args.se_init_bias,
            },
            "use_se": args.use_se,
            "se_reduction": args.se_reduction,
            "se_stages": list(se_stages),
            "se_init_bias": args.se_init_bias,
            "class_names": list(data.class_names),
        },
    )

    plot_training_curves(history, output_dir / "training_curves.png")

    # 使用验证集上表现最好的权重做最终测试集评估；筛选阶段可用 --skip-test 跳过。
    best_checkpoint_path = output_dir / "best.pt"
    if not args.skip_test:
        if best_checkpoint_path.exists():
            best_checkpoint = load_checkpoint(best_checkpoint_path, map_location=device)
            model.load_state_dict(best_checkpoint["model_state"])
            best_val_acc = float(best_checkpoint.get("best_val_acc", best_val_acc))
            best_epoch = int(best_checkpoint.get("best_epoch", best_epoch))
        model.to(device)
        model.eval()
        print(
            f"Load best checkpoint: epoch={best_epoch} | "
            f"best_val_acc={best_val_acc:.4f}"
        )

        metrics = run_test_evaluation(
            model=model,
            test_loader=data.test,
            raw_test_dataset=data.test_raw,
            device=device,
            output_dir=output_dir,
            class_names=data.class_names,
            criterion=criterion,
            amp=args.amp,
            max_batches=max_batches,
        )
    else:
        metrics = None
        print(
            f"[Info] --skip-test 已开启，跳过测试集评估；"
            f"best_val_acc={best_val_acc:.4f} (epoch {best_epoch})"
        )
        stale_test_files = [
            name
            for name in ("test_metrics.json", "test_predictions.npz")
            if (output_dir / name).exists()
        ]
        if stale_test_files:
            print(
                f"[Warn] 当前目录已有旧的 {', '.join(stale_test_files)}，"
                "本次 --skip-test 不会更新它们，请不要把它们当作本模型的测试结果。"
            )

    summary = {
        "model": args.model,
        "use_se": args.use_se,
        "se_reduction": args.se_reduction,
        "se_stages": list(se_stages),
        "se_init_bias": args.se_init_bias,
        "num_parameters": int(count_parameters(model, trainable_only=False)),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "test_skipped": bool(args.skip_test),
        "test_accuracy": None,
        "test_macro_f1": None,
        "device": str(device),
    }
    if metrics is not None:
        summary["test_accuracy"] = metrics["accuracy"]
        summary["test_macro_f1"] = metrics["macro_f1"]
    save_json(output_dir / "run_summary.json", summary)

    print("=" * 60)
    print("训练完成")
    print(f"  最佳验证准确率 : {best_val_acc:.4f} (epoch {best_epoch})")
    if metrics is not None:
        print(f"  测试集准确率   : {metrics['accuracy']:.4f}")
        print(f"  测试集 Macro-F1: {metrics['macro_f1']:.4f}")
    else:
        print("  测试集评估     : 已跳过（--skip-test）")
    print(f"  结果目录       : {output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
