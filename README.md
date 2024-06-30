# ViT_AlexNet_CIFRA100

## Bird Classification on CUB_200_2011

This repo includes a final project of _DATA620004, School of Data Sciences, Fudan University_.

### Project Introduction
This project compares two different kinds of image classification models based on Transformer and CNN. We implemented the two models with the help of `torchvision.models.vit_b_16` and `torchvision.models.alexnet`. The dataset we experimented on is CIFAR-100, which includes 100 classes of images and 600 images (500 for training and 100 for testing) in every class. We downloaded the dataset by `torchvision.datasets.CIFAR100`.

### Codes
file tree
```
│  inference.py
│  cnn_cutmix.ipynb
│  cnn_pretrained_cutmix.ipynb
│  cnn.ipynb
│  README.md
│  model.py
│  train.py
│  vit.ipynb
│  vit_cutmix.ipynb
│
├─data
│
└─models
│
└─runs
```

`model.py`: Load model from `torchvision.models`

`train.py`: Implement the code for training

`inference.py`: Code for inference

The Jupyterbooks show the procedure of loading and training models.