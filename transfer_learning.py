import datetime
import argparse
import torchvision
from lightly.transforms import utils
from lightly.data import LightlyDataset
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from models.heads.classifier import Classifier
import torch.nn as nn


# Shared Params
num_workers = 8
train_batch_size = 128
test_batch_size = 100
seed = 1
max_epochs = 200
train_root_dir = "./data/cifar10/train/"
test_root_dir = "./data/cifar10/test/"
freeze_backbone = False
precision = 32

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        help="[imagenet, simclr, moco, swav, byol]",
    )
    parser.add_argument("--ckpt_path", default=None, type=str)
    return parser.parse_args()


def startswith_remove_keys(target_key, remove_keys):
    for remove_key in remove_keys:
        if target_key.startswith(remove_key):
            return True
    return False


def replace_keys(
    in_dict,
    source="backbone.",
    target="",
    remove_keys=[],
):
    out_dict = {}
    for key, val in in_dict.items():
        if not startswith_remove_keys(key, remove_keys):
            out_key = key.replace(source, target)
            out_dict[out_key] = val

    return out_dict


def main():
    args = parse_args()

    pl.seed_everything(seed)

    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=CIFAR_MEAN,
                std=CIFAR_STD,
            ),
        ]
    )

    test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=CIFAR_MEAN,
                std=CIFAR_STD,
            ),
        ]
    )

    dataset_train = LightlyDataset(input_dir=train_root_dir, transform=train_transforms)

    dataset_test = LightlyDataset(input_dir=test_root_dir, transform=test_transforms)

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    logger = TensorBoardLogger(
        save_dir="tb_logs/transfer",
        name=args.model_name,
        version=datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
    )

    resnet = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    backbone = nn.Sequential(*list(resnet.children())[:-1])

    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path)
        if args.model_name == "swav":
            remove_keys = ["projection_head", "prototypes"]
        elif args.model_name == "moco_yoko":
            remove_keys = [
                "projection_head",
                "backbone_momentum",
                "projection_head_momentum",
            ]
        elif args.model_name == "moco":
            remove_keys = []
        elif args.model_name == "byol":
            remove_keys = [
                "projection_head",
                "prediction_head",
                "backbone_momentum",
                "projection_head_momentum",
            ]
        elif args.model_name == "simclr":
            remove_keys = ["projection_head"]
        else:
            raise Exception("Invalid model type")
        state_dict = replace_keys(ckpt["state_dict"], remove_keys=remove_keys)
        backbone.load_state_dict(state_dict)

    classifier = Classifier(backbone, max_epochs, freeze_backbone=freeze_backbone)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=1,
        accelerator="gpu",
        logger=logger,
        precision=precision,
        deterministic=True,
        callbacks=[LearningRateMonitor(logging_interval="step")],
    )
    trainer.fit(classifier, dataloader_train, dataloader_test)


if __name__ == "__main__":
    main()
