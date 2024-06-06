import torch
from torch import nn


# Input img - > Hidden dim -> mean, std -> Parameterization trick -> Decoder -> Output img
class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super().__init__()

        # encoder
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)

        # decoder
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        h = self.relu(self.img_2hid(x))
        print(h.shape)
        mu, sigma = self.hid_2mu(h), self.hid_2sigma(h)

        return mu, sigma

    def decode(self, z):
        h = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(h))

    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_rep = mu + sigma * epsilon
        x_rec = self.decode(z_rep)
        return x_rec, mu, sigma


if __name__ == "__main__":
    x = torch.randn(4, 28 * 28)
    vae = VariationalAutoencoder(input_dim=784)
    x_rec, mu, sigma = vae(x)
    print(x_rec.shape)
    print(mu.shape)
    print(sigma.shape)
