"""通用工具函数：随机种子、设备选择、指标累计与 JSON/CSV 读写。"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """固定 Python / NumPy / PyTorch 的随机种子，保证实验可复现。

    Args:
        seed: 随机种子。
        deterministic: 是否强制 cuDNN 使用确定性算法。开启后速度可能变慢，
            但同一设备上的结果更稳定。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    """为每个 DataLoader worker 设置独立且可复现的随机种子。

    该函数必须是模块级函数，才能在 Windows 的 spawn 多进程模式下被序列化。
    """
    del worker_id  # torch.initial_seed() 已经包含 worker 编号
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def resolve_device(name: str = "auto") -> torch.device:
    """解析命令行传入的设备字符串。"""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "当前环境没有可用的 CUDA 设备，请使用 --device cpu 或安装 GPU 版 PyTorch。"
        )
    return device


class AverageMeter:
    """滑动平均累计器，用于统计 loss / accuracy。"""

    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """统计模型参数量。"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def ensure_dir(path: str | Path) -> Path:
    """创建目录（若不存在）并返回 Path 对象。"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, obj: Any) -> None:
    """将对象保存为 UTF-8 JSON 文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    """读取 UTF-8 JSON 文件。"""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint(path: str | Path, map_location: Any = "cpu") -> dict:
    """加载本项目自己生成的 checkpoint。

    PyTorch 2.6 起 ``torch.load`` 默认 ``weights_only=True``，这里显式关闭
    以兼容包含优化器状态、参数配置等内容的 checkpoint。仅加载可信文件。
    """
    return torch.load(path, map_location=map_location, weights_only=False)
