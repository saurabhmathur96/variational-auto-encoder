import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
  if type(m) == nn.Linear:
      torch.nn.init.xavier_uniform_(m.weight)
      m.bias.data.fill_(0.01)

def binomial_loss(x, x_decoded, t_mu, t_log_sigma2):
    ''' Computes -ELBO for the case where p(x|t) is a continuous mixture of binomials  
      dKL(q, p) + log p(x|t) 
    '''
    # x: batch_size x #pixels
    # x_encoded: batch_size x #pixels
    # t_mu: batch_size x latent_size
    # t_log_sigma2: batch_size x latent_size
    dkl =  -0.5*torch.sum(1+t_log_sigma2 -t_mu.pow(2) - t_log_sigma2.exp(), 1)
    bce = torch.sum(F.binary_cross_entropy_with_logits(x_decoded, x, reduction='none'), 1)
    return torch.mean(bce + dkl)


class Encoder(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(Encoder,self).__init__()
    self.output_size = output_size
    self.backbone = nn.Sequential(
      nn.Linear(input_size, hidden_size),
      nn.ReLU(),
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(),
      nn.Linear(hidden_size, output_size*2)
    )
  def forward(self, x):
    output = self.backbone(x)
    return output[:, :self.output_size], output[:, self.output_size:]


def Decoder(input_size, hidden_size, output_size):
  return nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size)
  )
