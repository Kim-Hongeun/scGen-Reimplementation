from functools import reduce
import torch
from torch import nn
from torch.nn import functional

import pdb
class Encoder(nn.Module):
        
    def __init__(self, n_input, n_hidden1, n_hidden2, n_latent):
        super(Encoder, self).__init__()

        self.encoder_fc = nn.Sequential(nn.Linear(n_input, n_hidden1),
                                        #nn.BatchNorm1d(n_hidden1),
                                        nn.Dropout(0.3),
                                        nn.LeakyReLU(),
                                        nn.Linear(n_hidden1, n_hidden2),
                                        #nn.BatchNorm1d(n_hidden2),
                                        nn.Dropout(0.3),
                                        nn.LeakyReLU())
        
        self.mean_encoder = nn.Linear(n_hidden2, n_latent)
        self.var_encoder = nn.Linear(n_hidden2, n_latent)        

    def forward(self, x):
        y = self.encoder_fc(x)
        y_mean = self.mean_encoder(y)
        y_var = self.var_encoder(y)
                    
        return y_mean, y_var

class Decoder(nn.Module):
            
    def __init__(self, n_latent, n_hidden2, n_hidden1, n_output):
        super(Decoder, self).__init__()
        self.decoder_fc = nn.Sequential(nn.Linear(n_latent, n_hidden2),
                                        #nn.BatchNorm1d(n_hidden1),
                                        nn.Dropout(0.3),
                                        nn.LeakyReLU(),
                                        nn.Linear(n_hidden2, n_hidden1),
                                        #nn.BatchNorm1d(n_hidden2),
                                        nn.Dropout(0.3),
                                        nn.LeakyReLU(),
                                        nn.Linear(n_hidden1, n_output)
                                        #nn.BatchNorm1d(n_output),
                                        #nn.LeakyReLU()
                                        )
        
        
    def forward(self, x):
        y = self.decoder_fc(x)           
        return y

class scgenVAE(nn.Module):
    def __init__(self, Encoder, Decoder):
        
        super(scgenVAE, self).__init__()   
        self.Encoder = Encoder
        self.Decoder = Decoder

        #self.BCE_loss = nn.BCELoss(reduction='sum')

    def sample(self, mean, var):
        epsilon = torch.randn_like(var)
        z = mean + epsilon * var
        return z
        
    def loss_function(self, x_hat, x, mean, var):
        recon_loss = functional.mse_loss(x_hat, x, reduction='sum')
        KLD = - 0.5 * torch.sum(1+ var - mean.pow(2) - var.exp())
        return recon_loss + KLD    

    def forward(self, x):
        mean, var = self.Encoder(x)
        z = self.sample(mean, var)
        x_hat = self.Decoder(z)
        return x_hat, mean, var