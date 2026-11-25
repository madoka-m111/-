"""统计 SE 门控值，判断注意力是动态选择还是退化为静态通道缩放。

示例：
    python scripts/analyze_se.py --checkpoint outputs/se100_r8/best.pt --split val
    python scripts/analyze_se.py --checkpoint outputs/se100_r8/best.pt --split test ^
        --output outputs/se100_r8/se_gate_stats_test.json

输出字段：
    mean/std                 全部门控值的均值/标准差
    channel_std              每个通道均值在通道维上的标准差（静态通道偏置）
    sample_std               每个样本均值在样本维上的标准差（样本自适应成分）
    interaction_std          去掉通道/样本主效应后的交互项标准差
    fraction_gt_0.9          门控大于 0.9 的比例（饱和程度）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_dataloaders, build_test_loader
from src.model import SEBlock, build_model, parse_se_stages
from src.utils import load_checkpoint, resolve_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统计 SE 门控值（只读 checkpoint，不训练）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint 路径")
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="用于诊断的数据划分；默认 val，避免测试集泄漏争议",
    )
    parser.add_argument("--data-root", type=str, default="data", help="数据集根目录")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size")
    parser.add_argument(
        "--num-workers", type=int, default=0, help="DataLoader 工作进程数"
    )
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="数据划分随机种子")
    parser.add_argument("--download", action="store_true", help="本地无数据时自动下载")
    parser.add_argument("--device", type=str, default="auto", help="auto / cpu / cuda")
    parser.add_argument(
        "--max-batches", type=int, default=0, help="最多统计的 batch 数，0 表示不限制"
    )
    parser.add_argument(
        "--output", type=str, default="", help="输出 JSON；默认放在 checkpoint 同目录"
    )
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model_kwargs = checkpoint.get("model_kwargs") or {}
    use_se = bool(model_kwargs.get("use_se", checkpoint.get("use_se", False)))
    if not use_se:
        raise SystemExit("该 checkpoint 没有启用 SE，无法统计门控值。")

    se_reduction = int(
        model_kwargs.get("se_reduction", checkpoint.get("se_reduction", 16))
    )
    try:
        se_stages = parse_se_stages(
            model_kwargs.get("se_stages", checkpoint.get("se_stages", (1, 2, 3)))
        )
    except ValueError as exc:
        raise SystemExit(f"checkpoint 中的 se_stages 无效：{exc}") from exc
    se_init_bias = float(
        model_kwargs.get("se_init_bias", checkpoint.get("se_init_bias", 0.0))
    )
    class_names = checkpoint.get("class_names") or [str(i) for i in range(10)]

    model = build_model(
        str(checkpoint.get("model_name", "cifar_cnn")),
        num_classes=len(class_names),
        use_se=True,
        se_reduction=se_reduction,
        se_stages=se_stages,
        se_init_bias=se_init_bias,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    print(
        f"Checkpoint: {checkpoint_path} | SE stages: {se_stages} | "
        f"reduction: {se_reduction} | init bias: {se_init_bias} | "
        f"epoch: {checkpoint.get('epoch', 'unknown')}"
    )
    return model, checkpoint


def summarize_gates(gates: torch.Tensor, name: str) -> Dict[str, Any]:
    """gates shape: (num_samples, channels)。"""
    mean = gates.mean()
    std = gates.std(unbiased=False)
    channel_means = gates.mean(dim=0)
    sample_means = gates.mean(dim=1)
    interaction = (
        gates - channel_means.unsqueeze(0) - sample_means.unsqueeze(1) + mean
    )
    quantiles = torch.quantile(
        gates.flatten(),
        torch.tensor([0.01, 0.5, 0.99], dtype=gates.dtype, device=gates.device),
    )
    parts = name.split(".")
    stage = -1
    if len(parts) > 1 and parts[0] == "features":
        # features 由 [ConvBlock, MaxPool, ConvBlock, MaxPool, ...] 组成，
        # 因此 ConvBlock 的索引 0/2/4 对应 stage 1/2/3。
        stage = int(parts[1]) // 2 + 1
    return {
        "stage": stage,
        "module": name,
        "num_samples": int(gates.shape[0]),
        "num_channels": int(gates.shape[1]),
        "mean": float(mean.item()),
        "std": float(std.item()),
        "channel_std": float(channel_means.std(unbiased=False).item()),
        "sample_std": float(sample_means.std(unbiased=False).item()),
        "interaction_std": float(interaction.std(unbiased=False).item()),
        "q01": float(quantiles[0].item()),
        "median": float(quantiles[1].item()),
        "q99": float(quantiles[2].item()),
        "fraction_gt_0.9": float((gates > 0.9).float().mean().item()),
        "fraction_lt_0.1": float((gates < 0.1).float().mean().item()),
        "channel_mean_min": float(channel_means.min().item()),
        "channel_mean_median": float(channel_means.median().item()),
        "channel_mean_max": float(channel_means.max().item()),
        "fraction_channel_mean_gt_0.9": float(
            (channel_means > 0.9).float().mean().item()
        ),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device)

    if args.split == "val":
        data = build_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_ratio=args.val_ratio,
            seed=args.seed,
            download=args.download,
        )
        loader = data.val
    else:
        loader, _ = build_test_loader(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            download=args.download,
        )
    print(f"Split: {args.split} | samples: {len(loader.dataset):,}")

    captured: Dict[str, List[torch.Tensor]] = {}
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, SEBlock):

            def _hook(
                _module: torch.nn.Module,
                _inputs: Any,
                output: torch.Tensor,
                key: str = name,
            ) -> None:
                captured.setdefault(key, []).append(
                    output.detach().float().cpu()
                )

            handles.append(module.fc.register_forward_hook(_hook))

    if not handles:
        raise SystemExit("模型中没有找到 SEBlock，无法统计门控值。")

    num_batches = 0
    try:
        with torch.no_grad():
            progress = tqdm(loader, desc=f"SE gates ({args.split})", leave=False)
            for batch_idx, (inputs, _) in enumerate(progress):
                if args.max_batches > 0 and batch_idx >= args.max_batches:
                    break
                model(inputs.to(device, non_blocking=True))
                num_batches += 1
    finally:
        for handle in handles:
            handle.remove()

    if not captured:
        raise SystemExit("没有捕获到 SE 门控值，请检查 checkpoint 与模型结构。")

    stats = []
    for name, values in captured.items():
        gates = torch.cat(values, dim=0).squeeze(-1).squeeze(-1)
        stats.append(summarize_gates(gates, name))
    stats.sort(key=lambda item: (item["stage"], item["module"]))

    print("-" * 96)
    print(
        f"{'stage':>5} {'ch':>5} {'mean':>8} {'ch_std':>9} {'sample_std':>11} "
        f"{'inter_std':>10} {'>0.9':>7} {'ch_mean>0.9':>12}"
    )
    for item in stats:
        print(
            f"{item['stage']:>5} {item['num_channels']:>5} "
            f"{item['mean']:>8.4f} {item['channel_std']:>9.4f} "
            f"{item['sample_std']:>11.4f} {item['interaction_std']:>10.4f} "
            f"{item['fraction_gt_0.9']:>7.3f} "
            f"{item['fraction_channel_mean_gt_0.9']:>12.3f}"
        )
    print("-" * 96)
    print(
        "解读：channel_std >> sample_std 说明门控主要是静态通道缩放；"
        "mean 接近 1 且 >0.9 比例接近 1 说明该 stage 近似恒等。"
    )

    output_path = Path(args.output) if args.output else (
        Path(args.checkpoint).resolve().parent / f"se_gate_stats_{args.split}.json"
    )
    save_json(
        output_path,
        {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "split": args.split,
            "num_batches": num_batches,
            "num_samples": stats[0]["num_samples"] if stats else 0,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "se_stages": list(
                parse_se_stages(
                    (checkpoint.get("model_kwargs") or {}).get(
                        "se_stages", checkpoint.get("se_stages", (1, 2, 3))
                    )
                )
            ),
            "se_reduction": (checkpoint.get("model_kwargs") or {}).get(
                "se_reduction", checkpoint.get("se_reduction", 16)
            ),
            "se_init_bias": (checkpoint.get("model_kwargs") or {}).get(
                "se_init_bias", checkpoint.get("se_init_bias", 0.0)
            ),
            "stages": stats,
        },
    )
    print(f"结果已保存到：{output_path.resolve()}")


if __name__ == "__main__":
    main()
