import torch
import torchvision
import pytorch_lightning as pl
from lightly import loss
from pytorch_lightning import LightningModule, Trainer
from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.loss import NTXentLoss


class SimCLR(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()

        # create a ResNet backbone and remove the classification head

        self.backbone = backbone
        hidden_dim = 512
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)
        self.criterion = NTXentLoss()

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("loss/train", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 20)
        return optim
