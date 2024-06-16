import copy
import torch
import pytorch_lightning as pl
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import MoCoProjectionHead
from lightly.models.utils import (
    batch_shuffle,
    batch_unshuffle,
    deactivate_requires_grad,
    update_momentum,
)


class MocoModel(pl.LightningModule):
    def __init__(self, backbone, memory_bank_size: int, max_epochs: int):
        super().__init__()
        self.max_epochs = max_epochs

        self.backbone = backbone

        self.projection_head = MoCoProjectionHead(512, 512, 128)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = NTXentLoss(
            temperature=0.1, memory_bank_size=(memory_bank_size, 128)
        )

    def training_step(self, batch, batch_idx):
        (x_q, x_k), _, _ = batch

        update_momentum(self.backbone, self.backbone_momentum, 0.99)
        update_momentum(self.projection_head, self.projection_head_momentum, 0.99)

        q = self.backbone(x_q).flatten(start_dim=1)
        q = self.projection_head(q)

        k, shuffle = batch_shuffle(x_k)
        k = self.backbone_momentum(k).flatten(start_dim=1)
        k = self.projection_head_momentum(k)
        k = batch_unshuffle(k, shuffle)

        loss = self.criterion(q, k)
        self.log("loss/train", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(),
            lr=6e-2,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]
