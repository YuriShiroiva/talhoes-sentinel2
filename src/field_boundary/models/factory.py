from __future__ import annotations
from typing import Optional
import segmentation_models_pytorch as smp
from torch import nn


def build_model(
    architecture: str,
    encoder_name: str,
    encoder_weights,
    in_channels: int,
    classes: int = 1,
):
    """
    Cria modelos de segmentação usando segmentation-models-pytorch.

    Arquiteturas suportadas:
    - unet
    - unetplusplus
    - deeplabv3plus
    - fpn
    - pan
    """

    architecture = architecture.lower()

    arch_map = {
        "unet": "Unet",
        "unetplusplus": "UnetPlusPlus",
        "unet++": "UnetPlusPlus",
        "deeplabv3plus": "DeepLabV3Plus",
        "deeplabv3+": "DeepLabV3Plus",
        "fpn": "FPN",
        "pan": "PAN",
    }

    if architecture not in arch_map:
        raise ValueError(
            f"Arquitetura desconhecida: {architecture}. "
            f"Use uma das opções: {list(arch_map.keys())}"
        )

    return smp.create_model(
        arch=arch_map[architecture],
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,
    )
