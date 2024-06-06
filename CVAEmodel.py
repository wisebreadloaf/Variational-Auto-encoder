import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20
NUM_EPOCHS = 40
BATCH_SIZE = 1
LR_RATE = 3e-4

dataset = datasets.MNIST(
    root="dataset/", train=True, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=2, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1
        )
        self.fc1 = nn.Linear(64 * 30 * 30, H_DIM)
        self.fc_mu = nn.Linear(H_DIM, Z_DIM)
        self.fc_sigma = nn.Linear(H_DIM, Z_DIM)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(Z_DIM, H_DIM)
        self.fc3 = nn.Linear(H_DIM, 64 * 30 * 30)
        self.transconv2 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=2, stride=1, padding=1
        )
        self.transconv1 = nn.ConvTranspose2d(
            in_channels=32, out_channels=1, kernel_size=2, stride=1, padding=1
        )

    def encode(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        h = self.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_sigma(h)

    def reparameterize(self, mu, sigma):
        epsilon = torch.randn_like(sigma)
        return mu + sigma * epsilon

    def decode(self, z):
        z = self.relu(self.fc2(z))
        z = self.relu(self.fc3(z))
        z = z.view(-1, 64, 30, 30)
        z = self.relu(self.transconv2(z))
        z = self.transconv1(z)
        return torch.sigmoid(z)

    def forward(self, x):
        mu, sigma = self.encode(x)
        z = self.reparameterize(mu, sigma)
        return self.decode(z), mu, sigma
