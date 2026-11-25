"""加载训练好的 checkpoint，在 CIFAR-10 测试集上重新评估并生成报告素材。

示例：
    python evaluate.py --checkpoint outputs/final100/best.pt
    python evaluate.py --checkpoint outputs/final100/best.pt --output-dir outputs/final100_eval
"""

from __future__ import annotations

import argparse

import torch.nn as nn

from src.data import CIFAR10_CLASSES, build_test_loader
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在 CIFAR-10 测试集上评估训练好的模型。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="checkpoint 路径（best.pt / last.pt）"
    )
    parser.add_argument("--data-root", type=str, default="data", help="数据集根目录")
    parser.add_argument(
        "--output-dir", type=str, default="outputs/eval", help="评估结果输出目录"
    )
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")
    parser.add_argument(
        "--num-workers", type=int, default=4, help="DataLoader 工作进程数（Windows 可设 0）"
    )
    parser.add_argument(
        "--download", action="store_true", help="本地无数据时自动下载 CIFAR-10"
    )
    parser.add_argument("--device", type=str, default="auto", help="auto / cpu / cuda")
    parser.add_argument("--amp", action="store_true", help="使用混合精度推理（仅 CUDA）")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="最多评估的 batch 数，仅用于快速调试，0 表示不限制",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)

    if args.amp and device.type != "cuda":
        print("[Warn] --amp 仅在 CUDA 设备上生效，当前已自动关闭。")
        args.amp = False

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    class_names = tuple(checkpoint.get("class_names") or CIFAR10_CLASSES)
    model_name = str(checkpoint.get("model_name", "cifar_cnn"))

    # 从 checkpoint 恢复模型结构配置；旧 checkpoint 没有这些字段时使用基线默认值。
    model_kwargs = checkpoint.get("model_kwargs") or {}
    use_se = bool(model_kwargs.get("use_se", checkpoint.get("use_se", False)))
    se_reduction = int(
        model_kwargs.get("se_reduction", checkpoint.get("se_reduction", 16))
    )
    try:
        se_stages = parse_se_stages(
            model_kwargs.get(
                "se_stages", checkpoint.get("se_stages", (1, 2, 3))
            )
        )
    except ValueError as exc:
        raise SystemExit(f"checkpoint 中的 se_stages 无效：{exc}") from exc
    se_init_bias = float(
        model_kwargs.get(
            "se_init_bias", checkpoint.get("se_init_bias", 0.0)
        )
    )

    model = build_model(
        model_name,
        num_classes=len(class_names),
        use_se=use_se,
        se_reduction=se_reduction,
        se_stages=se_stages,
        se_init_bias=se_init_bias,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    print(f"Checkpoint: {args.checkpoint}")
    print(
        f"Model: {model_name} | SE: {use_se} | "
        f"SE stages: {se_stages if use_se else 'n/a'} | "
        f"SE init bias: {se_init_bias} | "
        f"params: {count_parameters(model):,} | "
        f"epoch: {checkpoint.get('epoch', 'unknown')} | "
        f"best_val_acc: {checkpoint.get('best_val_acc', 'unknown')}"
    )

    test_loader, raw_test_dataset = build_test_loader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
    )
    print(f"Test samples: {len(test_loader.dataset)}")

    save_json(
        output_dir / "eval_args.json",
        {
            **vars(args),
            "model_name": model_name,
            "use_se": use_se,
            "se_reduction": se_reduction,
            "se_stages": list(se_stages),
            "se_init_bias": se_init_bias,
            "class_names": list(class_names),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_best_val_acc": checkpoint.get("best_val_acc"),
        },
    )

    criterion = nn.CrossEntropyLoss()
    metrics = run_test_evaluation(
        model=model,
        test_loader=test_loader,
        raw_test_dataset=raw_test_dataset,
        device=device,
        output_dir=output_dir,
        class_names=class_names,
        criterion=criterion,
        amp=args.amp,
        max_batches=args.max_batches if args.max_batches > 0 else None,
    )

    print("=" * 60)
    print(f"测试集准确率   : {metrics['accuracy']:.4f}")
    print(f"测试集 Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"结果目录       : {output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
