import sys

import torch
from torchvision import transforms, datasets


train_loader = torch.utils.data.DataLoader(
  datasets.MNIST('data', train=True, download=True,
    transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
  ])),
batch_size=256, shuffle=True)
