"""比较两个 run 的逐样本预测：准确率差、McNemar 精确检验与配对 bootstrap。

使用前提：两个 run 目录下都有 train.py / evaluate.py 生成的
``test_predictions.npz``（包含 targets、preds、probs，可选 class_names）。

示例：
    python scripts/compare_runs.py --run-a outputs/final100_eval --run-b outputs/se100_r8_eval
    python scripts/compare_runs.py --run-a outputs/final100_eval --run-b outputs/se100_r8_eval ^
        --output outputs/compare_final100_vs_se.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="比较两个 run 的逐样本测试预测。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-a",
        type=str,
        required=True,
        help="基线 run 目录或 test_predictions.npz 路径",
    )
    parser.add_argument(
        "--run-b",
        type=str,
        required=True,
        help="对比 run 目录或 test_predictions.npz 路径",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        help="配对 bootstrap 次数，0 表示跳过置信区间",
    )
    parser.add_argument("--seed", type=int, default=42, help="bootstrap 随机种子")
    parser.add_argument(
        "--output", type=str, default="", help="可选：把比较结果保存为 JSON"
    )
    return parser.parse_args()


def resolve_predictions_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():
        path = path / "test_predictions.npz"
    if not path.exists():
        raise SystemExit(
            f"找不到预测文件：{path}\n"
            "请先运行 train.py 或 evaluate.py；新版评估会生成 test_predictions.npz。"
        )
    return path


def load_predictions(path: str | Path) -> Dict[str, Any]:
    path = resolve_predictions_path(path)
    with np.load(path, allow_pickle=False) as data:
        required = {"targets", "preds"}
        missing = required.difference(data.files)
        if missing:
            raise SystemExit(f"{path} 缺少字段：{sorted(missing)}")
        targets = np.asarray(data["targets"], dtype=np.int64).reshape(-1)
        preds = np.asarray(data["preds"], dtype=np.int64).reshape(-1)
        class_names: List[str] = []
        if "class_names" in data.files:
            class_names = [str(name) for name in data["class_names"].tolist()]
    if targets.shape != preds.shape:
        raise SystemExit(f"{path} 的 targets 与 preds 长度不一致。")
    return {
        "path": path,
        "targets": targets,
        "preds": preds,
        "class_names": class_names,
    }


def mcnemar_exact(n01: int, n10: int) -> float:
    """双侧精确 McNemar 检验。

    n01: A 错误且 B 正确；n10: A 正确且 B 错误。
    在 H0 下两者服从 Binomial(n01+n10, 0.5)。
    """
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def paired_bootstrap_ci(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> Tuple[float, float] | None:
    """对 (acc_b - acc_a) 做配对 bootstrap，返回 95% CI（百分点）。"""
    if n_bootstrap <= 0:
        return None
    n = len(correct_a)
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_bootstrap, dtype=np.float64)
    chunk = 200
    for start in range(0, n_bootstrap, chunk):
        size = min(chunk, n_bootstrap - start)
        indices = rng.integers(0, n, size=(size, n))
        diffs[start : start + size] = (
            correct_b[indices].mean(axis=1) - correct_a[indices].mean(axis=1)
        )
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(low * 100.0), float(high * 100.0)


def per_class_comparison(
    targets: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    class_names: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    class_ids = sorted(set(targets.tolist()))
    for class_id in class_ids:
        mask = targets == class_id
        name = (
            class_names[class_id]
            if 0 <= class_id < len(class_names)
            else f"class_{class_id}"
        )
        acc_a = float((preds_a[mask] == class_id).mean())
        acc_b = float((preds_b[mask] == class_id).mean())
        result[name] = {
            "support": int(mask.sum()),
            "accuracy_a": acc_a,
            "accuracy_b": acc_b,
            "diff_pp": (acc_b - acc_a) * 100.0,
        }
    return result


def main() -> None:
    args = parse_args()
    run_a = load_predictions(args.run_a)
    run_b = load_predictions(args.run_b)

    if len(run_a["targets"]) != len(run_b["targets"]):
        raise SystemExit(
            "两个预测文件的样本数不同："
            f"{len(run_a['targets'])} vs {len(run_b['targets'])}。"
            "请确认使用同一测试集且没有 --max-batches。"
        )
    if not np.array_equal(run_a["targets"], run_b["targets"]):
        raise SystemExit("两个预测文件的 targets 顺序/内容不同，无法做配对比较。")

    targets = run_a["targets"]
    preds_a = run_a["preds"]
    preds_b = run_b["preds"]
    correct_a = preds_a == targets
    correct_b = preds_b == targets

    acc_a = float(correct_a.mean())
    acc_b = float(correct_b.mean())
    diff_pp = (acc_b - acc_a) * 100.0
    n01 = int(np.sum(~correct_a & correct_b))
    n10 = int(np.sum(correct_a & ~correct_b))
    p_value = mcnemar_exact(n01, n10)
    ci = paired_bootstrap_ci(
        correct_a, correct_b, n_bootstrap=args.n_bootstrap, seed=args.seed
    )

    class_names = run_a["class_names"] or run_b["class_names"]
    per_class = per_class_comparison(targets, preds_a, preds_b, class_names)

    print("=" * 68)
    print(f"A: {run_a['path']}")
    print(f"B: {run_b['path']}")
    print(f"样本数: {len(targets):,}")
    print(f"A accuracy: {acc_a:.4%}")
    print(f"B accuracy: {acc_b:.4%}")
    print(f"B - A     : {diff_pp:+.3f} pp")
    if ci is not None:
        print(f"95% CI    : [{ci[0]:+.3f}, {ci[1]:+.3f}] pp")
    print(
        f"McNemar   : n01={n01} (A错B对), n10={n10} (A对B错), "
        f"exact p={p_value:.4g}"
    )
    print("-" * 68)
    print(f"{'class':>12} {'support':>8} {'A acc':>9} {'B acc':>9} {'diff':>9}")
    for name, item in per_class.items():
        print(
            f"{name:>12} {item['support']:>8d} "
            f"{item['accuracy_a']:>8.2%} {item['accuracy_b']:>8.2%} "
            f"{item['diff_pp']:>+8.2f}pp"
        )
    print("=" * 68)

    payload = {
        "run_a": str(run_a["path"]),
        "run_b": str(run_b["path"]),
        "num_samples": int(len(targets)),
        "accuracy_a": acc_a,
        "accuracy_b": acc_b,
        "diff_pp": diff_pp,
        "bootstrap_n": int(args.n_bootstrap),
        "ci95_pp": list(ci) if ci is not None else None,
        "mcnemar": {"n01": n01, "n10": n10, "p_value": float(p_value)},
        "per_class": per_class,
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到：{output_path.resolve()}")


if __name__ == "__main__":
    main()
