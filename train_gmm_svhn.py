from math import prod

import torch
from pythae.models.base.base_model import BaseDecoder, BaseEncoder, ModelOutput
from torch import nn

from multivae.data.datasets import MnistSvhn
from multivae.metrics.likelihoods.likelihoods import LikelihoodsEvaluator
from multivae.metrics.likelihoods.likelihoods_config import LikelihoodsEvaluatorConfig
from multivae.models import MMVAE, MMVAEConfig, MMVAEPlus, MMVAEPlusConfig, MVAE, MVAEConfig
from multivae.models import MVTCAE, MVTCAEConfig, MoPoE, MoPoEConfig
from multivae.models.base import BaseMultiVAEConfig

from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt

import utils

import numpy as np
from sklearn.mixture import GaussianMixture

gm = GaussianMixture(n_components=2, random_state=0).fit(X)
gm.means_
gm.predict([[0, 0], [12, 3]])


# Classes
class EncoderSVHN(BaseEncoder):
    def __init__(self, latent_dim):
        super().__init__()
        dataSize = torch.Size([3, 32, 32])
        imgChans = dataSize[0]
        fBase = 32  # base size of filter channels
        self.latent_dim = latent_dim

        self.enc = nn.Sequential(
            # input size: 3 x 32 x 32
            nn.Conv2d(imgChans, fBase, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase) x 16 x 16
            nn.Conv2d(fBase, fBase * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 8
            nn.Conv2d(fBase * 2, fBase * 4, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 4 x 4
        )
        self.c1 = nn.Conv2d(fBase * 4, self.latent_dim, 4, 1, 0, bias=True)
        self.c2 = nn.Conv2d(fBase * 4, self.latent_dim, 4, 1, 0, bias=True)
        # c1, c2 size: latent_dim x 1 x 1

    def forward(self, x):
        e = self.enc(x)
        return e


class DecoderSVHN(BaseDecoder):
    """Generate a SVHN image given a sample from the latent space."""

    def __init__(self, latent_dim):
        super().__init__()
        dataSize = torch.Size([3, 32, 32])
        imgChans = dataSize[0]
        fBase = 32  # base size of filter channels
        self.latent_dim = latent_dim
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, fBase * 4, 4, 1, 0, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 4 x 4
            nn.ConvTranspose2d(fBase * 4, fBase * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 8
            nn.ConvTranspose2d(fBase * 2, fBase, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase) x 16 x 16
            nn.ConvTranspose2d(fBase, imgChans, 4, 2, 1, bias=True),
            nn.Sigmoid()
            # Output size: 3 x 32 x 32
        )

    def forward(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
        out = self.dec(z.view(-1, *z.size()[-3:]))
        out = out.view(*z.size()[:-3], *out.size()[1:])
        # consider also predicting the length scale
        return out


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        base_channel = 64
        self.lin_in_dim = 2*2*base_channel*8

        # define encoder block
        self.encoder = EncoderSVHN(latent_dim)

        self.lin1 = nn.Sequential(
            nn.Linear(self.lin_in_dim, latent_dim),
            nn.ReLU(),
        )

        # linear layers for mu and logvar prediction
        self.lin11 = nn.Linear(latent_dim, latent_dim)
        self.lin12 = nn.Linear(latent_dim, latent_dim)

        # decoder block
        self.decoder = DecoderSVHN(latent_dim)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def encode(self, x):
        z = self.encoder(x)
        z = z.view(-1, self.lin_in_dim)
        z = self.lin1(z)
        mu = self.lin11(z)
        logvar = self.lin12(z)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar


# Dataset
DATA_PATH = './MNIST-SVHN'

train_set = MnistSvhn(data_path = DATA_PATH, split="train", data_multiplication=1, download=True)
test_set = MnistSvhn(data_path = DATA_PATH, split="train", data_multiplication=1, download=True)

B = 256
train_dataloader = DataLoader(train_set, batch_size=B, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=B, shuffle=False)


def compute_loss(inputs, outputs, mu, logvar):
    reconstruction_loss = nn.MSELoss(reduction='sum')(inputs, outputs)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    return 0*kl_loss + reconstruction_loss


def train_vae():

    epochs = 100
    latent_dimension = 100
    patience = 10

    device = torch.device('cuda:0') \
        if torch.cuda.is_available() \
        else torch.device('cpu')
    print('before VAE')

    # checkpoint = torch.load("./logs/VAE_SVHN_checkpoint.pth")

    model = VAE(latent_dimension).to(device)
    # model.load_state_dict(checkpoint['model'])
    
    optim = Adam(model.parameters(), lr=1e-4)
    # optim.load_state_dict(checkpoint['optim'])

    val_greater_count = 0
    print('starting the training')

    for e in range(epochs):
        running_loss = 0
        model.train()
        for batch in train_dataloader:
            images = batch['data']['svhn']
            images = images.to(device)
            model.zero_grad()
            outputs, mu, logvar = model(images)
            loss = compute_loss(images, outputs, mu, logvar)
            running_loss += loss
            loss.backward()
            optim.step()

        running_loss = running_loss/len(train_dataloader)
        model.eval()

        # save model
        torch.save({
            'epoch': e,
            'model': model.state_dict(),
            'running_loss': running_loss,
            'optim': optim.state_dict(),
        }, "./logs/VAE_SVHN_checkpoint_bce.pth")
        print("Epoch: {} Train Loss: {}".format(e+1, running_loss.item()))

        # check early stopping condition
        if val_greater_count >= patience:
            break

    z = torch.normal(0, 1, (64, latent_dimension)).to(device)
    images = model.decoder(z).cpu().detach().numpy()
    plt.imshow(utils.viz_array_grid(images, 8, 8))
    plt.savefig(f"VAE_SVHN_bce.pdf")

if __name__ == "__main__":
    train_vae()