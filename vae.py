import torch
import torch.nn as nn

class BinomialLoss(nn.Module):
  ''' Computes -ELBO for the case where p(x|t) is a continuous mixture of binomials '''
  def __init__(self):
    super(BinomialLoss,self).__init__()
  def forward(self, x, x_decoded, t_mu, t_log_sigma2):
    ''' dKL(q, p) + log p(x|t) '''
    # x: batch_size x #pixels
    # x_encoded: batch_size x #pixels
    # t_mu: batch_size x latent_size
    # t_log_sigma2: batch_size x latent_size
    pass
