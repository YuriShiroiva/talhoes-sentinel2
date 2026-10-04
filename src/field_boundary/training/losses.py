from __future__ import annotations
import torch
from torch import nn
import segmentation_models_pytorch as smp
import torch.nn.functional as F 

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)

class TverskyLoss(nn.Module):
    """
    Tversky Loss para segmentação binária.

    alpha controla penalização de falsos positivos.
    beta controla penalização de falsos negativos.

    Se beta > alpha, a loss penaliza mais falsos negativos,
    ajudando o modelo a não perder objetos pequenos.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        probs = probs.view(probs.shape[0], -1)
        targets = targets.view(targets.shape[0], -1)

        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1 - targets)).sum(dim=1)
        fn = ((1 - probs) * targets).sum(dim=1)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        return 1 - tversky.mean()


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss.

    A versão focal aumenta o peso dos exemplos difíceis.
    É útil quando há objetos pequenos e forte desbalanceamento.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 0.75,
        smooth: float = 1e-7,
    ):
        super().__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        tversky_loss = self.tversky(logits, targets)
        return torch.pow(tversky_loss, self.gamma)


class BCEFocalTverskyLoss(nn.Module):
    """
    Combinação de BCE + Focal Tversky.

    BCE mantém estabilidade pixel a pixel.
    Focal Tversky ajuda em objetos pequenos e exemplos difíceis.
    """

    def __init__(
        self,
        bce_weight: float = 0.3,
        focal_tversky_weight: float = 0.7,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 0.75,
    ):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.focal_tversky = FocalTverskyLoss(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        self.bce_weight = bce_weight
        self.focal_tversky_weight = focal_tversky_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.bce_weight * self.bce(logits, targets)
            + self.focal_tversky_weight * self.focal_tversky(logits, targets)
        )


def build_loss(
    loss_name: str,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    focal_tversky_weight: float = 0.7,
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float = 0.75,
):
    """
    Fábrica de losses para facilitar experimentos.
    """
    loss_name = loss_name.lower()

    if loss_name == "bce_dice":
        return BCEDiceLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
        )

    if loss_name == "tversky":
        return TverskyLoss(
            alpha=alpha,
            beta=beta,
        )

    if loss_name == "focal_tversky":
        return FocalTverskyLoss(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

    if loss_name == "bce_focal_tversky":
        return BCEFocalTverskyLoss(
            bce_weight=bce_weight,
            focal_tversky_weight=focal_tversky_weight,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

    raise ValueError(f"Loss desconhecida: {loss_name}")