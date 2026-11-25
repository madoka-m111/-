"""CIFAR-10 数据加载、训练/验证集划分与数据增强。

数据集划分策略：
    * 官方训练集 50,000 张 -> 训练集 45,000 / 验证集 5,000（默认 val_ratio=0.1）
    * 官方测试集 10,000 张 -> 仅在最终评估时使用
    * 采用分层抽样（stratified split），保证验证集各类别比例与训练集一致
    * 划分由固定随机种子控制，并将索引保存到 output_dir/split_indices.json

数据增强：
    * 训练：RandomCrop(32, padding=4) + RandomHorizontalFlip
    * 验证/测试：仅 ToTensor + Normalize，不做任何随机变换
    * 可通过 build_dataloaders(use_augment=False) 关闭训练增强，
      用于数据增强对比实验
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .utils import worker_init_fn

CIFAR10_CLASSES: Tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# CIFAR-10 训练集的通道均值与标准差
CIFAR10_MEAN: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: Tuple[float, float, float] = (0.2470, 0.2435, 0.2616)


class IndexedDataset(Dataset):
    """在原数据集上应用指定索引与变换。

    这样可以让训练子集与验证子集共享同一份原始数据，但使用不同的
    transform（训练集做数据增强，验证集不做），避免重复加载数据。
    """

    def __init__(
        self,
        dataset: Dataset,
        indices: Sequence[int],
        transform: transforms.Compose | None = None,
    ) -> None:
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image, label = self.dataset[self.indices[index]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


@dataclass
class CIFAR10Data:
    """训练、验证、测试 DataLoader 及原始测试集。"""

    train: DataLoader
    val: DataLoader
    test: DataLoader
    train_indices: List[int]
    val_indices: List[int]
    test_raw: Dataset
    class_names: Tuple[str, ...] = CIFAR10_CLASSES


def build_eval_transform() -> transforms.Compose:
    """验证/测试集变换：只做张量化与标准化。"""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )


def build_train_transform(use_augment: bool = True) -> transforms.Compose:
    """训练集变换：默认随机裁剪 + 随机水平翻转 + 标准化。

    Args:
        use_augment: 是否启用随机增强；False 时仅做 ToTensor + Normalize，
            用于“无增强 vs 有增强”的对照实验。
    """
    if not use_augment:
        return build_eval_transform()
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )


def split_train_val_indices(
    targets: Sequence[int],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """按类别分层划分训练集与验证集索引。

    Args:
        targets: 每个训练样本的类别标签。
        val_ratio: 验证集比例，默认 0.1。
        seed: 随机种子。

    Returns:
        (train_indices, val_indices)
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio 必须在 (0, 1) 之间，当前为 {val_ratio}")

    targets_array = np.asarray(targets)
    rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []

    for class_id in np.unique(targets_array):
        class_indices = np.flatnonzero(targets_array == class_id)
        rng.shuffle(class_indices)

        num_val = int(round(len(class_indices) * val_ratio))
        # 每个类别至少保留 1 个验证样本（当类别样本数足够时）
        if num_val == 0 and len(class_indices) > 1:
            num_val = 1

        val_indices.extend(class_indices[:num_val].tolist())
        train_indices.extend(class_indices[num_val:].tolist())

    if not train_indices or not val_indices:
        raise ValueError("划分后训练集或验证集为空，请检查 val_ratio 设置。")

    return train_indices, val_indices


def _load_cifar10(root: Path, train: bool, download: bool) -> datasets.CIFAR10:
    """加载 CIFAR-10，并在数据缺失时给出可操作的错误提示。"""
    try:
        return datasets.CIFAR10(
            root=str(root), train=train, download=download, transform=None
        )
    except RuntimeError as exc:
        split = "训练集" if train else "测试集"
        raise RuntimeError(
            f"未找到 CIFAR-10 {split}，请添加 --download 参数自动下载，"
            f"或手动将 cifar-10-batches-py 放到 {root / 'cifar-10-batches-py'}。"
        ) from exc


def build_test_loader(
    data_root: str | Path = "data",
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = False,
    pin_memory: bool = True,
) -> Tuple[DataLoader, Dataset]:
    """构建测试集 DataLoader 以及不做变换的原始测试集。

    返回的原始测试集用于可视化误分类样本（__getitem__ 返回 PIL 图像）。
    """
    root = Path(data_root)
    base_test = _load_cifar10(root, train=False, download=download)
    test_dataset = IndexedDataset(
        base_test,
        list(range(len(base_test))),
        transform=build_eval_transform(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
        persistent_workers=num_workers > 0,
    )
    return test_loader, base_test


def build_dataloaders(
    data_root: str | Path = "data",
    batch_size: int = 128,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    seed: int = 42,
    download: bool = False,
    pin_memory: bool = True,
    use_augment: bool = True,
) -> CIFAR10Data:
    """构建训练、验证、测试 DataLoader。

    Args:
        use_augment: 训练集是否启用 RandomCrop + RandomHorizontalFlip；
            False 时训练集仅做 ToTensor + Normalize。

    Notes:
        Windows 下如果 DataLoader 报多进程相关错误，可将 num_workers 设为 0。
    """
    root = Path(data_root)
    base_train = _load_cifar10(root, train=True, download=download)

    train_indices, val_indices = split_train_val_indices(
        targets=base_train.targets, val_ratio=val_ratio, seed=seed
    )

    train_dataset = IndexedDataset(
        base_train,
        train_indices,
        transform=build_train_transform(use_augment=use_augment),
    )
    val_dataset = IndexedDataset(
        base_train, val_indices, transform=build_eval_transform()
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    common_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
        persistent_workers=num_workers > 0,
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        generator=train_generator,
        **common_kwargs,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **common_kwargs)
    test_loader, base_test = build_test_loader(
        data_root=root,
        batch_size=batch_size,
        num_workers=num_workers,
        download=download,
        pin_memory=pin_memory,
    )

    return CIFAR10Data(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        train_indices=train_indices,
        val_indices=val_indices,
        test_raw=base_test,
    )
