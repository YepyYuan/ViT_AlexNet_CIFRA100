import torch
import torch.nn as nn

from torchvision.models import vit_b_16, alexnet
import torch
import torch.nn as nn


def load_model( net_struct = 'vit'):

    if net_struct == 'vit':
        model = vit_b_16()
        model.heads[0] = nn.Linear(model.heads[0].in_features, 100)

    else:
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2048)
        model.classifier.add_module (name='7', module=nn.ReLU(inplace=True))
        model.classifier.add_module (name='8', module=nn.Linear(2048, 1024))
        model.classifier.add_module (name='9', module=nn.ReLU(inplace=True))
        model.classifier.add_module (name='10', module=nn.Linear(1024, 100))

    return model