import torch
from lightly.loss import NTXentLoss
import torch.nn as nn
from lightly.models import ResNetGenerator
import copy
import pytorch_lightning as pl
from lightly.models.modules.heads import MoCoProjectionHead
from lightly.models.utils import (
    batch_shuffle,
    batch_unshuffle,
    deactivate_requires_grad,
    update_momentum,
)


class MocoModel(pl.LightningModule):
    def __init__(self, memory_bank_size: int, max_epochs: int):
        super().__init__()
        self.max_epochs = max_epochs

        # create a ResNet backbone and remove the classification head
        resnet = ResNetGenerator("resnet-18", 1, num_splits=8)
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )

        # create a moco model based on ResNet
        self.projection_head = MoCoProjectionHead(512, 512, 128)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        # create our loss with the optional memory bank
        self.criterion = NTXentLoss(
            temperature=0.1, memory_bank_size=(memory_bank_size, 128)
        )

    def training_step(self, batch, batch_idx):
        (x_q, x_k), _, _ = batch

        # update momentum
        update_momentum(self.backbone, self.backbone_momentum, 0.99)
        update_momentum(self.projection_head, self.projection_head_momentum, 0.99)

        # get queries
        q = self.backbone(x_q).flatten(start_dim=1)
        q = self.projection_head(q)

        # get keys
        k, shuffle = batch_shuffle(x_k)
        k = self.backbone_momentum(k).flatten(start_dim=1)
        k = self.projection_head_momentum(k)
        k = batch_unshuffle(k, shuffle)

        loss = self.criterion(q, k)
        self.log("train_loss_ssl", loss)
        return loss

    def on_train_epoch_end(self):
        self.custom_histogram_weights()

    # We provide a helper method to log weights in tensorboard
    # which is useful for debugging.
    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(),
            lr=6e-2,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]
