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

# Self-Supervised Learning
```
python3 contrastive_learning.py
```

# Fine-Tuning
modify <ckpt_path>
```
python3 transfer_learning.py
```

# TODO!
check difference of ckpt v1 and normal one
