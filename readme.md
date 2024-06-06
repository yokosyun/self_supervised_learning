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

```
python3 contrastive_learning.py --model_name swav
```

visualize log
```
tensorboard --logdir tb_logs/contrastive/
```


# Transfer Learning
```
python3 transfer_learning.py --model_name swav --ckpt_path <path-to.ckpt>
```
visualize log
```
tensorboard --logdir tb_logs/transfer/
```