import pytorch_lightning as pl
import torch
import torchvision

from lightly.data import LightlyDataset
from lightly.transforms import MoCoV2Transform, utils
from pytorch_lightning.loggers import TensorBoardLogger
from models.contrastive.moco import MocoModel

# Custom Params
memory_bank_size = 4096

# Shared Params
num_workers = 8
seed = 1
max_epochs = 100
batch_size = 512
path_to_train = "/home/yoko/data/cifar10/train/"
path_to_test = "/home/yoko/data/cifar10/test/"


def main():
    pl.seed_everything(seed)

    transform = MoCoV2Transform(
        input_size=32,
        gaussian_blur=0.0,
    )

    # We use the moco augmentations for training moco
    dataset_train_moco = LightlyDataset(input_dir=path_to_train, transform=transform)

    dataloader_train_moco = torch.utils.data.DataLoader(
        dataset_train_moco,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    logger = TensorBoardLogger("tb_logs", name="contrastive")

    model = MocoModel(memory_bank_size=memory_bank_size, max_epochs=max_epochs)
    trainer = pl.Trainer(
        max_epochs=max_epochs, devices=1, accelerator="gpu", logger=logger
    )
    trainer.fit(model, dataloader_train_moco)


if __name__ == "__main__":
    main()
