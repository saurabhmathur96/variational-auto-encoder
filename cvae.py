import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm



def train(train_loader, encoder, decoder, criterion, optimizer, class_count=10):
  losses = []
  for data, target in tqdm(train_loader):

    data = data.view(len(data), -1).float()
    target = F.one_hot(target, num_classes=class_count).float()

    optimizer.zero_grad()
    
    # encode
    mu, log_sigma2 = encoder(torch.cat([target, data], 1))

    # sample
    e = torch.randn_like(log_sigma2)
    t = mu + torch.exp(0.5*log_sigma2) * e

    # decode
    data_decoded = decoder(torch.cat([target, t], 1))


    loss = criterion(data, data_decoded, mu, log_sigma2)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()
  return losses
  
