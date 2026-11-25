"""估计“调度 × SE”的 2×2 交互效应。

固定数据增强开启，使用四个 run 的逐样本测试预测：

    A = no_sched100        （增强，无调度，无 SE）
    B = no_sched100_se     （增强，无调度，有 SE）
    C = final100           （增强，有调度，无 SE）
    D = se100_r8           （增强，有调度，有 SE）

SE 效应（无调度） = B - A
SE 效应（有调度） = D - C
交互             = (B - A) - (D - C)

示例：
    python scripts/analyze_interaction.py
    python scripts/analyze_interaction.py --output outputs/compare_interaction_se_sched.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="估计调度与 SE 的 2×2 交互效应。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--no-sched-base",
        type=str,
        default="outputs/no_sched100_eval",
        help="增强、无调度、无 SE 的评估目录（A）",
    )
    parser.add_argument(
        "--no-sched-se",
        type=str,
        default="outputs/no_sched100_se_eval",
        help="增强、无调度、有 SE 的评估目录（B）",
    )
    parser.add_argument(
        "--sched-base",
        type=str,
        default="outputs/final100_eval",
        help="增强、有调度、无 SE 的评估目录（C）",
    )
    parser.add_argument(
        "--sched-se",
        type=str,
        default="outputs/se100_r8_eval",
        help="增强、有调度、有 SE 的评估目录（D）",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=20000,
        help="联合配对 bootstrap 次数",
    )
    parser.add_argument("--seed", type=int, default=42, help="bootstrap 随机种子")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/compare_interaction_se_sched.json",
        help="结果 JSON 路径",
    )
    return parser.parse_args()


def resolve_predictions_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():
        path = path / "test_predictions.npz"
    if not path.exists():
        raise SystemExit(f"找不到预测文件：{path}")
    return path


def load_correct(path: str | Path) -> Tuple[np.ndarray, np.ndarray, Path]:
    resolved = resolve_predictions_path(path)
    with np.load(resolved, allow_pickle=False) as data:
        if "targets" not in data.files or "preds" not in data.files:
            raise SystemExit(f"{resolved} 缺少 targets 或 preds。")
        targets = np.asarray(data["targets"], dtype=np.int64).reshape(-1)
        preds = np.asarray(data["preds"], dtype=np.int64).reshape(-1)
    if targets.shape != preds.shape:
        raise SystemExit(f"{resolved} 的 targets 与 preds 长度不一致。")
    return targets, preds == targets, resolved


def bootstrap_interaction(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    correct_c: np.ndarray,
    correct_d: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> Tuple[float, float, float]:
    """返回交互的 95% CI（百分点）与双侧 bootstrap p 值。"""
    n = len(correct_a)
    rng = np.random.default_rng(seed)
    values = np.empty(n_bootstrap, dtype=np.float64)
    chunk = 200
    for start in range(0, n_bootstrap, chunk):
        size = min(chunk, n_bootstrap - start)
        indices = rng.integers(0, n, size=(size, n))
        effect_no_sched = (
            correct_b[indices].mean(axis=1) - correct_a[indices].mean(axis=1)
        )
        effect_sched = (
            correct_d[indices].mean(axis=1) - correct_c[indices].mean(axis=1)
        )
        values[start : start + size] = (effect_no_sched - effect_sched) * 100.0
    low, high = np.percentile(values, [2.5, 97.5])
    p_value = 2.0 * min(float((values <= 0).mean()), float((values >= 0).mean()))
    return float(low), float(high), min(1.0, p_value)


def main() -> None:
    args = parse_args()
    targets_a, correct_a, path_a = load_correct(args.no_sched_base)
    targets_b, correct_b, path_b = load_correct(args.no_sched_se)
    targets_c, correct_c, path_c = load_correct(args.sched_base)
    targets_d, correct_d, path_d = load_correct(args.sched_se)

    lengths = {len(targets_a), len(targets_b), len(targets_c), len(targets_d)}
    if len(lengths) != 1:
        raise SystemExit(f"四个预测文件的样本数不同：{sorted(lengths)}")
    for name, targets in [("B", targets_b), ("C", targets_c), ("D", targets_d)]:
        if not np.array_equal(targets_a, targets):
            raise SystemExit(f"{name} 的 targets 顺序/内容与 A 不同，无法配对。")

    acc_a = float(correct_a.mean())
    acc_b = float(correct_b.mean())
    acc_c = float(correct_c.mean())
    acc_d = float(correct_d.mean())
    se_no_sched = (acc_b - acc_a) * 100.0
    se_sched = (acc_d - acc_c) * 100.0
    interaction = se_no_sched - se_sched
    schedule_no_se = (acc_c - acc_a) * 100.0
    schedule_se = (acc_d - acc_b) * 100.0
    ci_low, ci_high, p_value = bootstrap_interaction(
        correct_a,
        correct_b,
        correct_c,
        correct_d,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    print("=" * 68)
    print(f"A (no_sched100)    : {path_a}  acc={acc_a:.4%}")
    print(f"B (no_sched100_se) : {path_b}  acc={acc_b:.4%}")
    print(f"C (final100)       : {path_c}  acc={acc_c:.4%}")
    print(f"D (se100_r8)       : {path_d}  acc={acc_d:.4%}")
    print("-" * 68)
    print(f"SE effect, no sched : {se_no_sched:+.3f} pp")
    print(f"SE effect, sched    : {se_sched:+.3f} pp")
    print(f"Interaction         : {interaction:+.3f} pp")
    print(f"95% CI              : [{ci_low:+.3f}, {ci_high:+.3f}] pp")
    print(f"Bootstrap p         : {p_value:.4g} ({args.n_bootstrap} reps)")
    print("-" * 68)
    print(f"Schedule effect, no SE : {schedule_no_se:+.3f} pp")
    print(f"Schedule effect, SE    : {schedule_se:+.3f} pp")
    print("=" * 68)

    payload: Dict[str, Any] = {
        "design": "2x2 with augmentation on: A=no_sched100, B=no_sched100_se, C=final100, D=se100_r8",
        "runs": {
            "A_no_sched_base": str(path_a),
            "B_no_sched_se": str(path_b),
            "C_sched_base": str(path_c),
            "D_sched_se": str(path_d),
        },
        "num_samples": int(len(targets_a)),
        "accuracy": {"A": acc_a, "B": acc_b, "C": acc_c, "D": acc_d},
        "effects_pp": {
            "se_without_schedule": se_no_sched,
            "se_with_schedule": se_sched,
            "interaction": interaction,
            "schedule_without_se": schedule_no_se,
            "schedule_with_se": schedule_se,
        },
        "bootstrap": {
            "n": int(args.n_bootstrap),
            "seed": int(args.seed),
            "ci95_pp": [ci_low, ci_high],
            "p_two_sided": p_value,
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"结果已保存到：{output_path.resolve()}")


if __name__ == "__main__":
    main()

