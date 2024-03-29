import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm



def train(train_loader, encoder, decoder, criterion, optimizer):
  losses = []
  for data, *_ in tqdm(train_loader):
    data = data.view(len(data), -1).float()

    optimizer.zero_grad()
    
    # encode
    mu, log_sigma2 = encoder(data)
    
    # sample
    e = torch.randn_like(log_sigma2)
    t = mu + torch.exp(0.5*log_sigma2) * e

    # decode
    data_decoded = decoder(t)
    
    loss = criterion(data, data_decoded, mu, log_sigma2)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()
  return losses
  