# Dataset
## CIFAR10
1. Register Kaggle
2. [Download CIFAR-10 from Kaggle](https://www.kaggle.com/datasets/swaroopkml/cifar10-pngs-in-folders)
3. place dataset as bellow
```
data
  -cifar10
      -train
      -test
```

# Install Dependencies

```
pip install pip --upgrade
pip3 install -r requirements.txt
```

# Contrastive Learning
train model
```
python3 contrastive_learning.py --model_name swav
```
visualize log
```
tensorboard --logdir tb_logs/contrastive/
```


# Transfer Learning
train model
```
python3 transfer_learning.py --model_name swav --ckpt_path <path-to.ckpt> (--freeze_backbone)
```

visualize log
```
tensorboard --logdir tb_logs/transfer/
```

# Reference
- [lightly](https://github.com/lightly-ai/lightly/tree/master)
- [contrastive learning papers](https://medium.com/@shunsukeyokokawa/self-supervised-learning-summary-2a0adf37954a)
- [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)
