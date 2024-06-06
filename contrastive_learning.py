import argparse
import torch
from torch import nn
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from lightly.data import LightlyDataset
from lightly.transforms import MoCoV2Transform
from lightly.transforms.swav_transform import SwaVTransform

from models.contrastive.moco import MocoModel
from models.contrastive.swav import SwaV


# fixed params
path_to_train = "./data/cifar10/train/"
batch_size = 128
num_workers = 8
seed = 1
max_epochs = 100
precision = 16


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    return parser.parse_args()


def main():
    args = parse_args()

    pl.seed_everything(seed)

    resnet = torchvision.models.resnet18()
    backbone = nn.Sequential(*list(resnet.children())[:-1])

    if args.model_name == "moco":
        model = MocoModel(backbone, memory_bank_size=4096, max_epochs=max_epochs)
        transform = MoCoV2Transform(
            input_size=32,
            gaussian_blur=0.0,
        )
    elif args.model_name == "swav":
        model = SwaV(backbone)
        transform = SwaVTransform()

    dataset = LightlyDataset(input_dir=path_to_train, transform=transform)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    logger = TensorBoardLogger("tb_logs", name="contrastive")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=1,
        accelerator="gpu",
        logger=logger,
        precision=precision,
    )
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    main()
