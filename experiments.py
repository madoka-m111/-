# -*- coding: utf-8 -*-
"""
实验二 第 6、7 条：自设对照实验、性能测试与图表生成
运行:
    python experiments.py

输出:
    results/maze_raw.csv            迷宫实验原始数据
    results/weighted_raw.csv        加权地形实验原始数据
    results/alpha_beta_raw.csv      Alpha-Beta 访问顺序实验原始数据
    results/p4_raw.csv              综合题深度/顺序实验原始数据
    results/baseline.json           四个题目基准测试用例的输入输出
    results/summary.json            报告使用的汇总指标
    results/fig1_maze.png           图1 迷宫：启发函数 × 规模/障碍密度
    results/fig2_weighted.png       图2 加权地形：可采纳性与扩展节点
    results/fig3_alpha_beta.png     图3 Alpha-Beta：访问顺序对剪枝的影响
    results/fig4_p4_depth.png       图4 综合题：深度与顺序的影响
"""
import copy
import csv
import importlib.util
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# 图表中文字体（Windows 自带）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_module(module_name: str, filename: str):
    """按文件名导入模块"""
    spec = importlib.util.spec_from_file_location(module_name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P1 = load_module("problem1", "problem1.py")
P2 = load_module("problem2", "problem2.py")
P3 = load_module("problem3", "problem3.py")
P4 = load_module("problem4", "problem4.py")

HEURISTIC_CN = {
    "zero": "h=0(Dijkstra)",
    "manhattan": "曼哈顿",
    "euclidean": "欧氏",
    "manhattan_min": "曼哈顿×最小代价",
    "manhattan_weighted": "曼哈顿×2",
}


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def write_csv(name, rows):
    if not rows:
        return
    path = RESULTS / name
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[写出] {path}  ({len(rows)} 行)")


def save_figure(fig, name):
    path = RESULTS / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[写出] {path}")


# =====================================================================
# 对照实验一：迷宫规模 / 障碍密度 × 启发函数
# =====================================================================
def carve_monotone_path(grid, n, rng, fill):
    """从 (0,0) 到 (n-1,n-1) 挖一条只向下/向右的随机通路，保证有解。"""
    r = c = 0
    grid[0][0] = fill()
    while r < n - 1 or c < n - 1:
        if r == n - 1:
            c += 1
        elif c == n - 1:
            r += 1
        elif rng.random() < 0.5:
            r += 1
        else:
            c += 1
        grid[r][c] = fill()


def gen_maze(n, density, seed):
    rng = random.Random(seed)
    grid = [["X" if rng.random() < density else "." for _ in range(n)] for _ in range(n)]
    carve_monotone_path(grid, n, rng, lambda: ".")
    grid[0][0], grid[n - 1][n - 1] = "S", "E"
    return grid



def run_maze_experiments():
    sizes = [21, 31, 41, 51]
    densities = [0.05, 0.15, 0.25, 0.35]
    seeds = 12
    heuristics = ["zero", "euclidean", "manhattan"]
    rows = []

    for n in sizes:
        for density in densities:
            for seed in range(seeds):
                grid = gen_maze(n, density, seed)
                for h in heuristics:
                    best = None
                    times = []
                    for _ in range(3):  # 时间取 3 次中位数，节点数确定
                        path, st = P1.astar_maze_with_stats(
                            (0, 0), (n - 1, n - 1), grid, heuristic=h)
                        times.append(st["search_time"])
                        if best is None:
                            best = path, st
                    path, st = best
                    rows.append({
                        "n": n, "density": density, "seed": seed, "heuristic": h,
                        "found": int(bool(path)),
                        "closed": st["nodes_in_closed"],
                        "expanded": st["nodes_expanded"],
                        "visited": st["nodes_visited"],
                        "open_max": st["nodes_in_open"],
                        "time_ms": statistics.median(times),
                        "path_len": st["path_length"],
                        "cost": st["optimal_cost"],
                    })

    write_csv("maze_raw.csv", rows)

    agg_density = {}
    for h in heuristics:
        for d in densities:
            vals = [r["closed"] for r in rows
                    if r["heuristic"] == h and r["n"] == 31 and r["density"] == d]
            agg_density[(h, d)] = mean(vals)
    agg_size = {}
    for h in heuristics:
        for n in sizes:
            vals = [r["closed"] for r in rows
                    if r["heuristic"] == h and r["n"] == n and r["density"] == 0.25]
            agg_size[(h, n)] = mean(vals)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3))
    markers = {"zero": "o", "euclidean": "s", "manhattan": "^"}
    for h in heuristics:
        axes[0].plot(densities, [agg_density[(h, d)] for d in densities],
                     marker=markers[h], label=HEURISTIC_CN[h])
        axes[1].plot(sizes, [agg_size[(h, n)] for n in sizes],
                     marker=markers[h], label=HEURISTIC_CN[h])
    axes[0].set_xlabel("障碍密度")
    axes[0].set_ylabel("平均已扩展节点数（closed）")
    axes[0].set_title("固定 31×31，障碍密度的影响")
    axes[1].set_xlabel("网格边长 n")
    axes[1].set_ylabel("平均已扩展节点数（closed）")
    axes[1].set_title("固定障碍密度 0.25，规模的影响")
    for ax in axes:
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, "fig1_maze.png")

    representative = {}
    for h in heuristics:
        vals = [r for r in rows if r["heuristic"] == h
                and r["n"] == 31 and r["density"] == 0.25]
        representative[h] = {
            "closed": mean(r["closed"] for r in vals),
            "visited": mean(r["visited"] for r in vals),
            "open_max": mean(r["open_max"] for r in vals),
            "time_ms": mean(r["time_ms"] for r in vals),
            "path_len": mean(r["path_len"] for r in vals),
        }
    summary = {
        "density_31": {f"{h}|{d}": v for (h, d), v in agg_density.items()},
        "size_density025": {f"{h}|{n}": v for (h, n), v in agg_size.items()},
    }
    return rows, representative, summary


# =====================================================================
# 对照实验二：加权地形中启发函数的可采纳性与最优性
# =====================================================================
def gen_weighted(n, density, seed, low, high):
    rng = random.Random(seed)
    grid = [["X" if rng.random() < density else str(rng.randint(low, high))
             for _ in range(n)] for _ in range(n)]
    carve_monotone_path(grid, n, rng, lambda: str(rng.randint(low, high)))
    grid[0][0], grid[n - 1][n - 1] = "S", "E"
    return grid


WEIGHTED_COUNTEREXAMPLE = [
    ["S", "1", "50", "1", "50"],
    ["50", "1", "50", "1", "1"],
    ["1", "1", "50", "50", "1"],
    ["1", "50", "1", "50", "1"],
    ["1", "1", "1", "50", "E"],
]


def run_weighted_experiments():
    n, density, seeds = 21, 0.20, 25
    scenarios = {
        "A_高代价(2~9)": (2, 9),
        "B_混合(1~9)": (1, 9),
    }
    heuristics = ["zero", "manhattan", "manhattan_min", "manhattan_weighted"]
    rows = []

    cases = []
    for scenario, (low, high) in scenarios.items():
        for seed in range(seeds):
            cases.append((scenario, seed, gen_weighted(n, density, seed, low, high)))
    # 构造反例：最短步数路线穿过大量 50，绕行低代价路线更优
    cases.append(("C_构造反例", 0, WEIGHTED_COUNTEREXAMPLE))

    for scenario, seed, grid in cases:
        for h in heuristics:
            best = None
            times = []
            for _ in range(3):
                cost, st = P2.astar_weighted_with_stats(
                    (0, 0), (len(grid) - 1, len(grid[0]) - 1), grid, heuristic=h)
                times.append(st["search_time"])
                if best is None:
                    best = cost, st
            cost, st = best
            rows.append({
                "scenario": scenario, "seed": seed, "heuristic": h,
                "cost": cost,
                "found": int(cost >= 0),
                "closed": st["nodes_in_closed"],
                "expanded": st["nodes_expanded"],
                "open_max": st["nodes_in_open"],
                "time_ms": statistics.median(times),
                "path_len": st["path_length"],
            })

    # 以 Dijkstra(h=0) 的代价作为最优基准，计算每个种子的相对差距
    base = {(r["scenario"], r["seed"]): r["cost"] for r in rows if r["heuristic"] == "zero"}
    for r in rows:
        b = base[(r["scenario"], r["seed"])]
        r["gap"] = r["cost"] - b
        r["optimal"] = int(r["cost"] == b)
    write_csv("weighted_raw.csv", rows)

    scenario_names = list(scenarios) + ["C_构造反例"]
    agg = {}
    for scenario in scenario_names:
        for h in heuristics:
            vals = [r for r in rows if r["scenario"] == scenario and r["heuristic"] == h]
            agg[(scenario, h)] = {
                "closed": mean(r["closed"] for r in vals),
                "time_ms": mean(r["time_ms"] for r in vals),
                "optimal_rate": mean(r["optimal"] for r in vals) * 100,
                "mean_gap": mean(r["gap"] for r in vals),
                "max_gap": max(r["gap"] for r in vals),
            }

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3))
    width = 0.2
    x = range(len(scenario_names))
    colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"]
    for i, h in enumerate(heuristics):
        axes[0].bar([v + (i - 1.5) * width for v in x],
                    [agg[(s, h)]["closed"] for s in scenario_names],
                    width=width, label=HEURISTIC_CN[h], color=colors[i])
        axes[1].bar([v + (i - 1.5) * width for v in x],
                    [100 - agg[(s, h)]["optimal_rate"] for s in scenario_names],
                    width=width, label=HEURISTIC_CN[h], color=colors[i])
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(scenario_names, fontsize=7)
    axes[0].set_ylabel("平均已扩展节点数（closed）")
    axes[0].set_title("(a) 搜索开销")
    axes[0].set_yscale("log")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(scenario_names, fontsize=7)
    axes[1].set_ylabel("未找到最优解的样本比例 (%)")
    axes[1].set_title("(b) 最优性（相对 h=0）")
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, "fig2_weighted.png")

    return rows, {f"{s}|{h}": v for (s, h), v in agg.items()}



# =====================================================================
# 对照实验三：Alpha-Beta 的子节点访问顺序
# =====================================================================
DEFAULT_TREE = {
    "is_maximizing_player": True,
    "children": [
        {"is_maximizing_player": False, "children": [
            {"is_maximizing_player": True, "children": [{"value": 3}, {"value": 5}]},
            {"is_maximizing_player": True, "children": [{"value": 6}, {"value": 9}]},
        ]},
        {"is_maximizing_player": False, "children": [
            {"is_maximizing_player": True, "children": [{"value": 6}, {"value": 9}]},
            {"is_maximizing_player": True, "children": [{"value": 1}, {"value": 2}]},
        ]},
    ],
}


def count_tree_nodes(node):
    return 1 + sum(count_tree_nodes(c) for c in node.children)


def run_alpha_beta(root):
    stats = {"nodes_visited": 0, "leaf_nodes_visited": 0, "pruned_nodes": 0,
             "max_depth": 0, "search_time": 0,
             "total_nodes": count_tree_nodes(root)}
    t0 = time.perf_counter()
    value = P3.alpha_beta(root, float("-inf"), float("inf"), 0, [0], stats)
    stats["search_time"] = (time.perf_counter() - t0) * 1000
    return value, stats


def run_plain_minimax(root):
    stats = {"nodes_visited": 0, "leaf_nodes_visited": 0, "pruned_nodes": 0,
             "max_depth": 0, "search_time": 0,
             "total_nodes": count_tree_nodes(root)}
    t0 = time.perf_counter()
    value = P3.minimax_without_pruning(root, 0, [0], stats)
    stats["search_time"] = (time.perf_counter() - t0) * 1000
    return value, stats


def make_random_tree(depth, branch, rng, counter, is_max=True):
    node_id = next(counter)
    if depth == 0:
        return P3.GameState(node_id, value=rng.randint(0, 9), is_max=is_max)
    node = P3.GameState(node_id, is_max=is_max)
    for _ in range(branch):
        node.children.append(
            make_random_tree(depth - 1, branch, rng, counter, not is_max))
    return node


def shuffle_tree(node, rng):
    if node.children:
        rng.shuffle(node.children)
        for child in node.children:
            shuffle_tree(child, rng)


def plain_tree_value(node):
    if not node.children:
        return node.value
    values = [plain_tree_value(c) for c in node.children]
    return max(values) if node.is_max else min(values)


def order_oracle(node):
    """按子树真实 Minimax 值排序：MAX 降序、MIN 升序（剪枝上界）。"""
    if not node.children:
        return node
    scored = [(plain_tree_value(c), c) for c in node.children]
    scored.sort(key=lambda item: item[0], reverse=node.is_max)
    node.children = [c for _, c in scored]
    for child in node.children:
        order_oracle(child)
    return node


def shallow_value(node, depth=2):
    """不完整搜索的静态估计：向下看 depth 层，截断处取均值。"""
    if not node.children:
        return node.value
    values = [shallow_value(c, depth - 1) for c in node.children]
    if depth <= 1:
        return sum(values) / len(values)
    return max(values) if node.is_max else min(values)


def order_shallow(node):
    """按 2 层浅搜索值排序（不依赖完整搜索，可在线使用）。"""
    if not node.children:
        return node
    scored = [(shallow_value(c, 2), c) for c in node.children]
    scored.sort(key=lambda item: item[0], reverse=node.is_max)
    node.children = [c for _, c in scored]
    for child in node.children:
        order_shallow(child)
    return node

def run_alpha_beta_experiments():
    rng = random.Random(20261022)
    trials = 200
    strategies = ["原始顺序", "随机重排", "浅层排序", "最优排序"]
    rows = []

    for tree_id in range(trials):
        counter = iter(range(10 ** 6))
        tree = make_random_tree(4, 3, rng, counter)
        total_nodes = count_tree_nodes(tree)
        mm_value, mm_stats = run_plain_minimax(copy.deepcopy(tree))

        for strat in strategies:
            work = copy.deepcopy(tree)
            if strat == "随机重排":
                shuffle_tree(work, random.Random(90000 + tree_id))
            elif strat == "浅层排序":
                order_shallow(work)
            elif strat == "最优排序":
                order_oracle(work)
            value, stats = run_alpha_beta(work)
            assert value == mm_value, (tree_id, strat, value, mm_value)
            rows.append({
                "tree_id": tree_id, "strategy": strat,
                "visited": stats["nodes_visited"],
                "leaves": stats["leaf_nodes_visited"],
                "pruned": stats["pruned_nodes"],
                "time_ms": stats["search_time"],
                "value": value,
                "minimax_visited": mm_stats["nodes_visited"],
                "total_nodes": total_nodes,
                "saving_pct": (1 - stats["nodes_visited"] / mm_stats["nodes_visited"]) * 100,
            })

    write_csv("alpha_beta_raw.csv", rows)

    summary = {}
    for strat in strategies:
        vals = [r for r in rows if r["strategy"] == strat]
        summary[strat] = {
            "visited_mean": mean(r["visited"] for r in vals),
            "visited_median": statistics.median(r["visited"] for r in vals),
            "visited_min": min(r["visited"] for r in vals),
            "visited_max": max(r["visited"] for r in vals),
            "leaves_mean": mean(r["leaves"] for r in vals),
            "pruned_mean": mean(r["pruned"] for r in vals),
            "saving_pct": mean(r["saving_pct"] for r in vals),
            "time_ms": mean(r["time_ms"] for r in vals),
        }

    # 默认测试树（题目三给定用例）
    default_root = P3.build_tree_from_dict(DEFAULT_TREE)
    default_total = count_tree_nodes(default_root)
    default_mm_value, default_mm_stats = run_plain_minimax(copy.deepcopy(default_root))
    default_results = {}
    for strat in ["原始顺序", "随机重排", "浅层排序", "最优排序"]:
        work = copy.deepcopy(default_root)
        if strat == "随机重排":
            work.children = list(reversed(work.children))
        elif strat == "浅层排序":
            order_shallow(work)
        elif strat == "最优排序":
            order_oracle(work)
        value, stats = run_alpha_beta(work)
        default_results[strat] = {
            "value": value, "visited": stats["nodes_visited"],
            "leaves": stats["leaf_nodes_visited"],
            "pruned": stats["pruned_nodes"], "time_ms": stats["search_time"],
        }
    default_results["Minimax"] = {
        "value": default_mm_value, "visited": default_mm_stats["nodes_visited"],
        "leaves": default_mm_stats["leaf_nodes_visited"],
        "pruned": default_mm_stats["pruned_nodes"],
        "time_ms": default_mm_stats["search_time"],
    }

    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    data = [[r["visited"] for r in rows if r["strategy"] == s] for s in strategies]
    ax.boxplot(data, tick_labels=strategies, showfliers=False, widths=0.55)
    for i, s in enumerate(strategies, start=1):
        ax.scatter([i], [summary[s]["visited_mean"]], marker="D", color="red", zorder=3)
    random_total = count_tree_nodes(tree)
    ax.axhline(random_total, color="gray", linestyle="--", linewidth=1,
               label=f"完整树节点数 {random_total}")
    ax.set_ylabel("访问节点数（对数坐标）")
    ax.set_title("随机 3 叉 4 层树（200 棵）不同访问顺序的搜索开销")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, "fig3_alpha_beta.png")

    return rows, summary, default_results, {
        "total_nodes": default_total,
        "theoretical_best": 2 * 3 ** 2 - 1,
    }


# =====================================================================
# 对照实验四：综合题（寻宝博弈）的深度与子节点顺序
# =====================================================================
SCENARIO_A = {
    "name": "A_默认7×7",
    "map": [
        ["A", "1", "1", "X", "2", "1", "T1"],
        ["1", "X", "6", "X", "1", "X", "1"],
        ["1", "1", "T2", "1", "1", "1", "1"],
        ["X", "1", "X", "8", "X", "1", "X"],
        ["1", "1", "1", "1", "T3", "1", "1"],
        ["1", "X", "9", "X", "1", "X", "1"],
        ["T4", "1", "1", "X", "1", "1", "B"],
    ],
    "treasures": {
        "T1": {"pos": (0, 6), "value": 9},
        "T2": {"pos": (2, 2), "value": 6},
        "T3": {"pos": (4, 4), "value": 7},
        "T4": {"pos": (6, 0), "value": 7},
    },
    "start_a": (0, 0),
    "start_b": (6, 6),
}

SCENARIO_B = {
    "name": "B_自设9×9",
    "map": [
        ["A", "1", "1", "X", "1", "1", "1", "1", "T1"],
        ["1", "1", "1", "X", "1", "1", "1", "1", "1"],
        ["1", "T2", "1", "1", "1", "1", "X", "1", "1"],
        ["1", "1", "1", "X", "1", "1", "X", "1", "1"],
        ["1", "1", "1", "X", "1", "1", "T3", "1", "1"],
        ["1", "1", "1", "X", "1", "1", "X", "1", "1"],
        ["T4", "1", "1", "1", "1", "1", "X", "1", "1"],
        ["1", "1", "1", "X", "1", "1", "1", "1", "1"],
        ["1", "1", "1", "X", "1", "T5", "1", "1", "B"],
    ],
    "treasures": {
        "T1": {"pos": (0, 8), "value": 10},
        "T2": {"pos": (2, 1), "value": 5},
        "T3": {"pos": (4, 6), "value": 9},
        "T4": {"pos": (6, 0), "value": 6},
        "T5": {"pos": (8, 5), "value": 8},
    },
    "start_a": (0, 0),
    "start_b": (8, 8),
}


def p4_build(scenario, depth):
    values = {n: info["value"] for n, info in scenario["treasures"].items()}
    dist, points = P4.compute_distance_matrix(
        scenario["map"], scenario["start_a"], scenario["start_b"],
        scenario["treasures"])
    root = P4.build_game_tree(
        dist, depth, "A", set(values), "A", "B",
        0, 0, values, node_id=0)
    return root, dist, values, points


def p4_plain_value(node):
    if not node.children:
        return node.a_val - node.b_val
    values = [p4_plain_value(ch) for _, ch in node.children]
    return max(values) if node.player == "A" else min(values)


def p4_order_oracle(node):
    """按子局面真实 Minimax 值排序（MAX 降序 / MIN 升序）。"""
    if not node.children:
        return node
    scored = [(p4_plain_value(ch), t, ch) for t, ch in node.children]
    scored.sort(key=lambda item: item[0], reverse=(node.player == "A"))
    node.children = [(t, ch) for _, t, ch in scored]
    for _, ch in node.children:
        p4_order_oracle(ch)
    return node


def p4_order_gain(node, dist, values):
    """按“宝藏价值 - 移动代价”的即时净收益降序排序（在线可用）。"""
    if not node.children:
        return node
    scored = []
    for t, ch in node.children:
        pos = node.a_pos if node.player == "A" else node.b_pos
        scored.append((values[t] - dist[(pos, t)], t, ch))
    scored.sort(key=lambda item: item[0], reverse=True)
    node.children = [(t, ch) for _, t, ch in scored]
    for _, ch in node.children:
        p4_order_gain(ch, dist, values)
    return node


def p4_shuffle(node, rng):
    if node.children:
        rng.shuffle(node.children)
        for _, ch in node.children:
            p4_shuffle(ch, rng)



def p4_run_alpha_beta(root):
    stats = {"nodes_visited": 0, "leaf_nodes_visited": 0, "pruned_nodes": 0,
             "search_time": 0, "total_nodes": P4.count_nodes(root)}
    t0 = time.perf_counter()
    P4.alpha_beta(root, float("-inf"), float("inf"), stats)
    stats["search_time"] = (time.perf_counter() - t0) * 1000
    move, value = P4.find_best_move(root)
    return move, value, stats


def p4_run_minimax(root):
    stats = {"nodes_visited": 0, "leaf_nodes_visited": 0, "pruned_nodes": 0,
             "search_time": 0, "total_nodes": P4.count_nodes(root)}
    t0 = time.perf_counter()
    P4.minimax(root, stats)
    stats["search_time"] = (time.perf_counter() - t0) * 1000
    move, value = P4.find_best_move(root)
    return move, value, stats


def run_p4_experiments():
    scenarios = [SCENARIO_A, SCENARIO_B]
    rows = []
    depth_summary = {}
    distance_default = None

    for scenario in scenarios:
        max_h = len(scenario["treasures"])
        for depth in range(2, max_h + 1):
            base_root, dist, values, points = p4_build(scenario, depth)
            total_nodes = P4.count_nodes(base_root)

            mm_root = copy.deepcopy(base_root)
            mm_move, mm_value, mm_stats = p4_run_minimax(mm_root)

            ab_move, ab_value, ab_stats = p4_run_alpha_beta(copy.deepcopy(base_root))
            assert ab_value == mm_value

            rows.append({"scenario": scenario["name"], "depth": depth,
                         "strategy": "Minimax", "run": 0,
                         "visited": mm_stats["nodes_visited"],
                         "leaves": mm_stats["leaf_nodes_visited"],
                         "pruned": mm_stats["pruned_nodes"],
                         "time_ms": mm_stats["search_time"],
                         "value": mm_value, "best_move": mm_move,
                         "total_nodes": total_nodes, "saving_pct": 0.0})
            rows.append({"scenario": scenario["name"], "depth": depth,
                         "strategy": "AB原始", "run": 0,
                         "visited": ab_stats["nodes_visited"],
                         "leaves": ab_stats["leaf_nodes_visited"],
                         "pruned": ab_stats["pruned_nodes"],
                         "time_ms": ab_stats["search_time"],
                         "value": ab_value, "best_move": ab_move,
                         "total_nodes": total_nodes,
                         "saving_pct": (1 - ab_stats["nodes_visited"] / mm_stats["nodes_visited"]) * 100})

            random_runs = []
            for run in range(20):
                work = copy.deepcopy(base_root)
                p4_shuffle(work, random.Random(depth * 10000 + run))
                move, value, stats = p4_run_alpha_beta(work)
                assert value == mm_value
                random_runs.append((move, value, stats))
                rows.append({"scenario": scenario["name"], "depth": depth,
                             "strategy": "AB随机", "run": run,
                             "visited": stats["nodes_visited"],
                             "leaves": stats["leaf_nodes_visited"],
                             "pruned": stats["pruned_nodes"],
                             "time_ms": stats["search_time"],
                             "value": value, "best_move": move,
                             "total_nodes": total_nodes,
                             "saving_pct": (1 - stats["nodes_visited"] / mm_stats["nodes_visited"]) * 100})


            gain_root = copy.deepcopy(base_root)
            p4_order_gain(gain_root, dist, values)
            gain_move, gain_value, gain_stats = p4_run_alpha_beta(gain_root)
            assert gain_value == mm_value
            rows.append({"scenario": scenario["name"], "depth": depth,
                         "strategy": "AB增益排序", "run": 0,
                         "visited": gain_stats["nodes_visited"],
                         "leaves": gain_stats["leaf_nodes_visited"],
                         "pruned": gain_stats["pruned_nodes"],
                         "time_ms": gain_stats["search_time"],
                         "value": gain_value, "best_move": gain_move,
                         "total_nodes": total_nodes,
                         "saving_pct": (1 - gain_stats["nodes_visited"] / mm_stats["nodes_visited"]) * 100})

            oracle_root = copy.deepcopy(base_root)
            p4_order_oracle(oracle_root)
            oracle_move, oracle_value, oracle_stats = p4_run_alpha_beta(oracle_root)
            assert oracle_value == mm_value
            rows.append({"scenario": scenario["name"], "depth": depth,
                         "strategy": "AB最优排序", "run": 0,
                         "visited": oracle_stats["nodes_visited"],
                         "leaves": oracle_stats["leaf_nodes_visited"],
                         "pruned": oracle_stats["pruned_nodes"],
                         "time_ms": oracle_stats["search_time"],
                         "value": oracle_value, "best_move": oracle_move,
                         "total_nodes": total_nodes,
                         "saving_pct": (1 - oracle_stats["nodes_visited"] / mm_stats["nodes_visited"]) * 100})

            depth_summary[f"{scenario['name']}|H{depth}"] = {
                "total_nodes": total_nodes,
                "minimax": {"visited": mm_stats["nodes_visited"],
                            "value": mm_value, "move": mm_move,
                            "time_ms": mm_stats["search_time"]},
                "ab_original": {"visited": ab_stats["nodes_visited"],
                                "value": ab_value, "move": ab_move,
                                "time_ms": ab_stats["search_time"]},
                "ab_random_mean": mean(r[2]["nodes_visited"] for r in random_runs),
                "ab_random_min": min(r[2]["nodes_visited"] for r in random_runs),
                "ab_random_max": max(r[2]["nodes_visited"] for r in random_runs),
                "ab_gain": {"visited": gain_stats["nodes_visited"],
                            "value": gain_value, "move": gain_move,
                            "time_ms": gain_stats["search_time"]},
                "ab_oracle": {"visited": oracle_stats["nodes_visited"],
                              "value": oracle_value, "move": oracle_move,
                              "time_ms": oracle_stats["search_time"]},
            }

            if distance_default is None:
                keys = sorted(points)
                distance_default = {
                    "keys": keys,
                    "points": {k: list(points[k]) for k in keys},
                    "dist": {f"{k1}->{k2}": dist[(k1, k2)]
                             for k1 in keys for k2 in keys},
                }

    write_csv("p4_raw.csv", rows)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3))
    for ax, scenario in zip(axes, scenarios):
        hs = list(range(2, len(scenario["treasures"]) + 1))
        series = {
            "Minimax": [depth_summary[f"{scenario['name']}|H{h}"]["minimax"]["visited"] for h in hs],
            "AB原始": [depth_summary[f"{scenario['name']}|H{h}"]["ab_original"]["visited"] for h in hs],
            "AB增益排序": [depth_summary[f"{scenario['name']}|H{h}"]["ab_gain"]["visited"] for h in hs],
            "AB最优排序": [depth_summary[f"{scenario['name']}|H{h}"]["ab_oracle"]["visited"] for h in hs],
        }
        for label, vals in series.items():
            ax.plot(hs, vals, marker="o", label=label)
        ax.set_xticks(hs)
        ax.set_xlabel("搜索深度 H")
        ax.set_ylabel("访问节点数（对数坐标）")
        ax.set_title(scenario["name"])
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, "fig4_p4_depth.png")

    return rows, depth_summary, distance_default



# =====================================================================
# 四个题目给定/示例测试用例的基准结果
# =====================================================================
MAZE_SAMPLE = [
    ["S", ".", ".", "X", "."],
    [".", "X", ".", ".", "."],
    [".", ".", ".", "X", "."],
    ["X", ".", "X", ".", "."],
    [".", ".", ".", ".", "E"],
]

WEIGHTED_SAMPLE = [
    ["S", "1", "1", "8", "1"],
    ["1", "8", "3", "1", "1"],
    ["1", "1", "1", "8", "1"],
    ["8", "1", "8", "1", "1"],
    ["1", "1", "1", "1", "E"],
]


def weighted_path_detail(path, grid):
    total = 0
    detail = []
    for i, (r, c) in enumerate(path):
        cell = grid[r][c]
        cost = 0 if cell in ("S", "E") else int(cell)
        if i > 0:
            total += cost
        detail.append({"step": i, "pos": [r, c], "cell": cell,
                       "enter_cost": cost, "g": total})
    return detail


def run_baseline():
    baseline = {}

    p1_runs = {}
    for h in ["zero", "manhattan", "euclidean"]:
        path, st = P1.astar_maze_with_stats((0, 0), (4, 4), MAZE_SAMPLE, heuristic=h)
        p1_runs[h] = {
            "path": [list(p) for p in path],
            "path_len": st["path_length"], "cost": st["optimal_cost"],
            "closed": st["nodes_in_closed"], "expanded": st["nodes_expanded"],
            "visited": st["nodes_visited"], "open_max": st["nodes_in_open"],
            "time_ms": st["search_time"],
        }
    baseline["problem1"] = {
        "input": {"start": [0, 0], "end": [4, 4], "grid": MAZE_SAMPLE},
        "runs": p1_runs,
    }

    p2_runs = {}
    for h in ["zero", "manhattan", "manhattan_min"]:
        cost, st = P2.astar_weighted_with_stats(
            (0, 0), (4, 4), WEIGHTED_SAMPLE, heuristic=h)
        p2_runs[h] = {
            "cost": cost, "path": [list(p) for p in st["path"]],
            "path_len": st["path_length"], "closed": st["nodes_in_closed"],
            "visited": st["nodes_visited"], "time_ms": st["search_time"],
        }
    p2_runs["path_detail"] = weighted_path_detail(p2_runs["zero"]["path"], WEIGHTED_SAMPLE)
    baseline["problem2"] = {
        "input": {"start": [0, 0], "end": [4, 4], "grid": WEIGHTED_SAMPLE},
        "runs": p2_runs,
    }

    root = P3.build_tree_from_dict(DEFAULT_TREE)
    ab_value, ab_stats = run_alpha_beta(copy.deepcopy(root))
    mm_value, mm_stats = run_plain_minimax(copy.deepcopy(root))
    baseline["problem3"] = {
        "input": DEFAULT_TREE,
        "alpha_beta": {"value": ab_value, "visited": ab_stats["nodes_visited"],
                       "leaves": ab_stats["leaf_nodes_visited"],
                       "pruned": ab_stats["pruned_nodes"],
                       "time_ms": ab_stats["search_time"]},
        "minimax": {"value": mm_value, "visited": mm_stats["nodes_visited"],
                    "leaves": mm_stats["leaf_nodes_visited"],
                    "time_ms": mm_stats["search_time"]},
    }

    p4_runs = {}
    for depth in [2, 4]:
        r, dist, values, points = p4_build(SCENARIO_A, depth)
        move, value, st = p4_run_alpha_beta(r)
        mm_move, mm_value, mm_st = p4_run_minimax(copy.deepcopy(r))
        p4_runs[f"H={depth}"] = {
            "best_move": move, "best_value": value,
            "alpha_beta": {"visited": st["nodes_visited"],
                           "leaves": st["leaf_nodes_visited"],
                           "pruned": st["pruned_nodes"],
                           "time_ms": st["search_time"]},
            "minimax": {"best_move": mm_move, "best_value": mm_value,
                        "visited": mm_st["nodes_visited"]},
        }
    baseline["problem4"] = {
        "input": {k: (v if k != "treasures" else
                      {n: {"pos": list(info["pos"]), "value": info["value"]}
                       for n, info in v.items()})
                  for k, v in SCENARIO_A.items() if k != "name"},
        "runs": p4_runs,
    }

    with open(RESULTS / "baseline.json", "w", encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False, indent=2)
    print(f"[写出] {RESULTS / 'baseline.json'}")
    return baseline



# =====================================================================
# 主程序
# =====================================================================
def main():
    t_start = time.time()
    print("=" * 72)
    print(" 实验二 第6/7条：自设对照实验与图表生成（不修改四个题目代码）")
    print("=" * 72)

    print("\n[1/5] 四个题目基准用例 ...")
    baseline = run_baseline()

    print("\n[2/5] 迷宫：规模 / 障碍密度 × 启发函数 ...")
    maze_rows, maze_rep, maze_summary = run_maze_experiments()

    print("\n[3/5] 加权地形：启发函数可采纳性 ...")
    weighted_rows, weighted_summary = run_weighted_experiments()

    print("\n[4/5] Alpha-Beta：随机树 + 访问顺序 ...")
    ab_rows, ab_summary, ab_default, ab_meta = run_alpha_beta_experiments()

    print("\n[5/5] 综合题：深度 + 访问顺序 ...")
    p4_rows, p4_summary, p4_distance = run_p4_experiments()

    summary = {
        "maze_representative_31x31_d025": maze_rep,
        "maze_aggregate": maze_summary,
        "weighted": weighted_summary,
        "alpha_beta_random_trees": ab_summary,
        "alpha_beta_default_tree": ab_default,
        "alpha_beta_meta": ab_meta,
        "p4_depth": p4_summary,
        "p4_distance_default": p4_distance,
        "runtime_sec": time.time() - t_start,
    }
    with open(RESULTS / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[写出] {RESULTS / 'summary.json'}")

    # ---------- 控制台汇总 ----------
    print("\n" + "=" * 72)
    print(" 基准用例结果")
    print("=" * 72)
    for h, run in baseline["problem1"]["runs"].items():
        print(f" P1 {h:<10} 路径长度={run['path_len']:>2}  "
              f"closed={run['closed']:>3}  时间={run['time_ms']:.3f} ms")
        print(f"    路径: {run['path']}")
    for h, run in baseline["problem2"]["runs"].items():
        if h == "path_detail":
            continue
        print(f" P2 {h:<12} 总代价={run['cost']:>3}  closed={run['closed']:>3}  "
              f"时间={run['time_ms']:.3f} ms")
    print(f" P2 代价累计: {[(d['pos'], d['enter_cost'], d['g']) for d in baseline['problem2']['runs']['path_detail']]}")
    p3 = baseline["problem3"]
    print(f" P3 Alpha-Beta 值={p3['alpha_beta']['value']}  访问={p3['alpha_beta']['visited']}  "
          f"剪枝={p3['alpha_beta']['pruned']}  |  Minimax 访问={p3['minimax']['visited']}")
    for key, run in baseline["problem4"]["runs"].items():
        print(f" P4 {key}: best={run['best_move']} value={run['best_value']}  "
              f"AB访问={run['alpha_beta']['visited']}  Minimax访问={run['minimax']['visited']}")

    print("\n" + "=" * 72)
    print(" 迷宫代表结果（31×31，障碍密度 0.25，12 个随机种子均值）")
    print("=" * 72)
    for h, s in maze_rep.items():
        print(f" {h:<10} closed={s['closed']:8.1f}  visited={s['visited']:8.1f}  "
              f"open_max={s['open_max']:7.1f}  时间={s['time_ms']:7.3f} ms  "
              f"路径长度={s['path_len']:.1f}")

    print("\n" + "=" * 72)
    print(" 加权地形汇总（21×21，25 个随机种子）")
    print("=" * 72)
    for key, s in weighted_summary.items():
        print(f" {key:<32} closed={s['closed']:8.1f}  最优率={s['optimal_rate']:5.1f}%  "
              f"平均差距={s['mean_gap']:5.2f}  最大差距={s['max_gap']:>3}  "
              f"时间={s['time_ms']:7.3f} ms")

    print("\n" + "=" * 72)
    print(" Alpha-Beta 访问顺序汇总（200 棵随机 3 叉 4 层树）")
    print("=" * 72)
    for strat, s in ab_summary.items():
        print(f" {strat:<6} 访问均值={s['visited_mean']:6.1f}  中位数={s['visited_median']:5.0f}  "
              f"范围=[{s['visited_min']},{s['visited_max']}]  "
              f"节省={s['saving_pct']:5.1f}%  剪枝次数={s['pruned_mean']:.2f}")

    print("\n" + "=" * 72)
    print(" 综合题深度汇总")
    print("=" * 72)
    for key, s in p4_summary.items():
        print(f" {key:<18} 总节点={s['total_nodes']:>3}  "
              f"AB原始={s['ab_original']['visited']:>3}  "
              f"随机={s['ab_random_mean']:6.1f}  "
              f"增益={s['ab_gain']['visited']:>3}  最优={s['ab_oracle']['visited']:>3}  "
              f"Minimax={s['minimax']['visited']:>3}  "
              f"best={s['ab_original']['move']}({s['ab_original']['value']})")

    print(f"\n全部实验完成，用时 {time.time() - t_start:.1f} s。结果目录: {RESULTS}")
    return summary


if __name__ == "__main__":
    main()

