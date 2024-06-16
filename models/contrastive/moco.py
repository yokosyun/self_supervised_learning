import torch
from torch import nn
import pytorch_lightning as pl
from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
import copy


class Moco(pl.LightningModule):
    def __init__(self, backbone, feature_dim=512, projection_dim=128, temperature=0.07):
        super().__init__()
        self.backbone = backbone
        self.projection_head = MoCoProjectionHead(
            feature_dim, projection_dim, projection_dim
        )
        self.register_buffer("queue", torch.randn(projection_dim, 4096))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.queue_ptr = torch.zeros(1, dtype=torch.long)
        self.temperature = temperature
        self.criterion = NTXentLoss(temperature=self.temperature)

        # Deep copy the backbone to create the key_encoder
        self.key_encoder = copy.deepcopy(self.backbone)
        self.key_projection_head = MoCoProjectionHead(
            feature_dim, projection_dim, projection_dim
        )

        # Momentum encoder update weight
        self.moco_momentum = 0.99

    def forward(self, x):
        # Feature extraction part
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x)
        return nn.functional.normalize(x, dim=1)

    def training_step(self, batch, batch_idx):
        # Assuming the batch structure is ((imgs1, imgs2), labels) or more complex
        # Adjust the unpacking logic based on the printed structure
        (imgs1, imgs2), *_ = batch  # Ignore additional elements like labels

        imgs1, imgs2 = imgs1.to(self.device), imgs2.to(self.device)

        # Query and key representations
        q1 = self.forward(imgs1)
        q2 = self.forward(imgs2)

        with torch.no_grad():
            # Update key encoder using momentum
            self._momentum_update_key_encoder()

            # Compute key representations
            k1 = self.key_encoder(imgs2).flatten(start_dim=1)
            k1 = self.key_projection_head(k1)
            k2 = self.key_encoder(imgs1).flatten(start_dim=1)
            k2 = self.key_projection_head(k2)

        # Ensure tensors are 2D before computing loss
        q1 = q1.view(q1.size(0), -1)
        q2 = q2.view(q2.size(0), -1)
        k1 = k1.view(k1.size(0), -1)
        k2 = k2.view(k2.size(0), -1)

        # Compute contrastive loss
        loss = self.criterion(q1, k2) + self.criterion(q2, k1)
        self.log("loss/train", loss, on_step=True, on_epoch=True)

        return loss

    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(
            self.backbone.parameters(), self.key_encoder.parameters()
        ):
            param_k.data = param_k.data * self.moco_momentum + param_q.data * (
                1.0 - self.moco_momentum
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def dequeue_and_enqueue(self, keys):
        """Update the queue."""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue.size(1) % batch_size == 0  # Ensure clean division

        # Replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue.size(1)  # move pointer
        self.queue_ptr[0] = ptr
