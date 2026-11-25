"""CIFAR-10 分类模型：带 BatchNorm、Dropout 与全局平均池化的 VGG 风格 CNN。

网络结构（输入 3 x 32 x 32）：
    Stage 1: 2 x [Conv3x3(3->64) + BN + ReLU] -> MaxPool -> 64 x 16 x 16
    Stage 2: 2 x [Conv3x3(64->128) + BN + ReLU] -> MaxPool -> 128 x 8 x 8
    Stage 3: 2 x [Conv3x3(128->256) + BN + ReLU] -> MaxPool -> 256 x 4 x 4
    Head:    GlobalAvgPool -> Dropout(0.5) -> Linear(256 -> 10)

可选结构：
    * 每个 ConvBlock 后可以插入 SE（Squeeze-and-Excitation）通道注意力，
      通过 ``--use-se`` 开启，默认关闭以保证与基线权重兼容；
    * 可用 ``--se-stages`` 选择只在部分 stage 插入 SE，便于做结构消融；
    * 可用 ``--se-init-bias`` 调整 SE 最后一层 bias，使初始门控接近 1。

设计考虑：
    * 全部使用 3x3 小卷积核，两层卷积感受野更大且参数更少；
    * BatchNorm 加速收敛并起到一定正则化作用；
    * 每个 stage 后使用递增的 Dropout，分类头使用较高 Dropout；
    * 全局平均池化替代大尺寸全连接层，显著减少参数量并降低过拟合；
    * 可选 SE 模块用很低的参数成本做通道重标定，便于做注意力消融实验。
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn


def parse_se_stages(spec: str | Sequence[int] | None = None) -> Tuple[int, ...]:
    """解析 SE stage 选择，返回排序后的 stage 编号（1-based）。

    支持 ``"1,2,3"``、``"1, 2"``、``[1, 2]`` 或 ``None``（默认全部 stage）。
    这样 train.py、evaluate.py 和诊断脚本可以共享同一套校验逻辑。
    """
    if spec is None or spec == "":
        return (1, 2, 3)

    if isinstance(spec, str):
        parts = [part.strip() for part in spec.split(",") if part.strip()]
        if not parts:
            raise ValueError("--se-stages 不能为空，示例：--se-stages 1,2")
        try:
            stages = tuple(int(part) for part in parts)
        except ValueError as exc:
            raise ValueError(
                f"--se-stages 必须是逗号分隔的整数，当前为 {spec!r}"
            ) from exc
    else:
        stages = tuple(int(part) for part in spec)

    if not stages:
        raise ValueError("--se-stages 至少需要包含一个 stage。")
    if len(set(stages)) != len(stages):
        raise ValueError(f"--se-stages 不能包含重复值，当前为 {stages}")
    invalid = [stage for stage in stages if stage not in (1, 2, 3)]
    if invalid:
        raise ValueError(f"--se-stages 只能取 1/2/3，当前包含 {invalid}")
    return tuple(sorted(stages))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力模块。

    先对每个通道做全局平均池化（Squeeze），再用两层 1x1 卷积学习通道权重
    （Excitation），最后逐通道乘回原特征图。使用 1x1 卷积而不是全连接层，
    可以直接处理 NCHW 特征图，避免额外的 reshape。
    """

    def __init__(
        self, channels: int, reduction: int = 16, init_bias: float = 0.0
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("SEBlock 的 channels 必须为正整数。")
        if reduction < 1:
            raise ValueError(f"SE reduction 必须 >= 1，当前为 {reduction}")

        self.init_bias = float(init_bias)
        hidden_channels = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # 先使用默认初始化；CIFARCNN 完成全局 Kaiming 初始化后，会再调用
        # reset_parameters() 设置最后一层，使初始门控为 sigmoid(init_bias)。
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """零初始化最后一层权重，并按 init_bias 设置初始门控。

        init_bias=0.0 时初始门控为 0.5（旧行为）；设为 2.0 时约为 0.88，
        让 SE 在训练初期接近恒等映射，减少对已收敛基线的扰动。
        """
        final_conv = self.fc[2]
        nn.init.zeros_(final_conv.weight)
        nn.init.constant_(final_conv.bias, self.init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(self.pool(x))
        return x * scale


class ConvBlock(nn.Module):
    """两个 3x3 卷积 + BN + ReLU，可选 SE 通道注意力与 Dropout。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        use_se: bool = False,
        se_reduction: int = 16,
        se_init_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.se = (
            SEBlock(
                out_channels,
                reduction=se_reduction,
                init_bias=se_init_bias,
            )
            if use_se
            else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.se(self.block(x))
        return self.dropout(x)


class CIFARCNN(nn.Module):
    """适用于 CIFAR-10 的自定义卷积神经网络，可选 SE 通道注意力。"""

    def __init__(
        self,
        num_classes: int = 10,
        widths: Sequence[int] = (64, 128, 256),
        block_dropouts: Sequence[float] = (0.1, 0.2, 0.3),
        classifier_dropout: float = 0.5,
        use_se: bool = False,
        se_reduction: int = 16,
        se_stages: Sequence[int] | None = (1, 2, 3),
        se_init_bias: float = 0.0,
    ) -> None:
        super().__init__()
        if len(widths) != 3 or len(block_dropouts) != 3:
            raise ValueError("当前实现要求 widths 与 block_dropouts 均包含 3 个元素。")

        selected_stages = set(parse_se_stages(se_stages))
        self.use_se = bool(use_se)
        self.se_stages = tuple(sorted(selected_stages))
        self.se_reduction = int(se_reduction)
        self.se_init_bias = float(se_init_bias)

        in_channels = 3
        stages = []
        for stage_index, (out_channels, dropout) in enumerate(
            zip(widths, block_dropouts), start=1
        ):
            stages.append(
                ConvBlock(
                    in_channels,
                    out_channels,
                    dropout=dropout,
                    use_se=self.use_se and stage_index in selected_stages,
                    se_reduction=se_reduction,
                    se_init_bias=se_init_bias,
                )
            )
            stages.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels

        self.features = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(classifier_dropout),
            nn.Linear(in_channels, num_classes),
        )

        self._initialize_weights()
        if self.use_se:
            # 全局 Kaiming 初始化会覆盖 SE 的最后一层，这里再次设置，
            # 使训练开始时每个通道的门控为 sigmoid(se_init_bias)。
            self._initialize_se()

    def _initialize_se(self) -> None:
        """重新初始化 SE 模块的最后一层。"""
        for module in self.modules():
            if isinstance(module, SEBlock):
                module.reset_parameters()

    def _initialize_weights(self) -> None:
        """Kaiming 初始化卷积层，便于深层网络稳定训练。"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """返回全局池化前的特征图，便于 Grad-CAM 等可视化分析。"""
        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_model(
    name: str = "cifar_cnn",
    num_classes: int = 10,
    use_se: bool = False,
    se_reduction: int = 16,
    se_stages: Sequence[int] | None = (1, 2, 3),
    se_init_bias: float = 0.0,
) -> nn.Module:
    """模型工厂函数，后续可通过添加分支扩展更多架构。"""
    if name == "cifar_cnn":
        return CIFARCNN(
            num_classes=num_classes,
            use_se=use_se,
            se_reduction=se_reduction,
            se_stages=se_stages,
            se_init_bias=se_init_bias,
        )
    raise ValueError(f"未知模型名称：{name}，目前支持：['cifar_cnn']")
