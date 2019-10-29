import neural_net, vae, cvae

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.utils import save_image

train_loader = torch.utils.data.DataLoader(
  datasets.MNIST('data', train=True, download=True,
    transform=transforms.Compose([
      transforms.ToTensor()
  ])),
batch_size=128, shuffle=True)


def init_weights(m):
  if type(m) == nn.Linear:
      torch.nn.init.xavier_uniform_(m.weight)
      m.bias.data.fill_(0.01)

variant = sys.argv[1]
embedding_size = int(sys.argv[2])

if variant == 'vae':
  encoder = neural_net.Encoder(28*28, 256, embedding_size)
  encoder.apply(neural_net.init_weights)
  
  decoder = neural_net.Decoder(embedding_size, 256, 28*28)
  decoder.apply(neural_net.init_weights)
  
  criterion = neural_net.binomial_loss
  params = list(encoder.parameters()) + list(decoder.parameters())
  optimizer = torch.optim.Adam(params, lr=1e-3)

  t = torch.randn(100, embedding_size)

  for epoch in range(1, 10+1):
    print ('Epoch %d:' % epoch)
    losses = vae.train(train_loader, encoder, decoder, criterion, optimizer)
    print (np.mean(losses))
    with torch.no_grad():
      
      images = torch.sigmoid(decoder(t))
      save_image(images.view(100, 1, 28, 28), '%s-sampled%d.png' % (variant, epoch), nrow=10)

elif variant == 'cvae':
  encoder = neural_net.Encoder(10 + 28*28, 256, embedding_size)
  encoder.apply(neural_net.init_weights)
  
  decoder = neural_net.Decoder(10 + embedding_size, 256, 28*28)
  decoder.apply(neural_net.init_weights)
  
  criterion = neural_net.binomial_loss
  params = list(encoder.parameters()) + list(decoder.parameters())
  optimizer = torch.optim.Adam(params, lr=1e-3)

  t = torch.randn(100, embedding_size)
  c = torch.LongTensor(sum([[i]*10 for i in range(10)], []))

  for epoch in range(1, 10+1):
    print ('Epoch %d:' % epoch)
    losses = cvae.train(train_loader, encoder, decoder, criterion, optimizer)
    print (np.mean(losses))
    with torch.no_grad():
      
      images = torch.sigmoid(decoder(torch.cat([F.one_hot(c, num_classes=10).float(), t], 1)))
      save_image(images.view(100, 1, 28, 28), '%s-sampled%d.png' % (variant, epoch), nrow=10)
  
