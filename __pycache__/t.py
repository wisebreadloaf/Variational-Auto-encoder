import torch
import torch.nn as nn
from torchvision.utils import save_image
import torchvision.datasets as datasets
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20

NUM_EPOCHS = 40
BATCH_SIZE = 128
LR_RATE = 3e-4
dataset = datasets.MNIST(
    root="dataset/", train=True, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

loop = tqdm(enumerate(train_loader))

count = 0


def encoder(x):
    conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=2, stride=1, padding=1)
    relu = nn.ReLU()
    conv2 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=2, stride=1, padding=1)
    dense = nn.Linear(128 * 4 * 30 * 30, 200)

    output = conv1(x)
    output = relu(output)
    output = conv2(output)
    output = relu(output)
    print(output.shape)
    output = torch.flatten(output)
    print(output.shape)
    output = dense(output)
    print(output.shape)


def decoder(z):
    dense = nn.Linear(200, 128 * 4 * 30 * 30)
    relu = nn.ReLU()
    # torch.reshape(128, 4, 30, 30)
    transconv2 = nn.ConvTranspose2d(
        in_channels=4, out_channels=2, kernel_size=2, stride=1, padding=1
    )
    transconv1 = nn.ConvTranspose2d(
        in_channels=2, out_channels=1, kernel_size=2, stride=1, padding=1
    )

    output = dense(z.squeeze())
    output = relu(output)
    output = torch.reshape(128, 4, 30, 30)
    output = transconv2(output)
    output = relu(output)
    output = transconv1(output)
    print(output.shape)


for i, (x, _) in loop:
    if count < 1:
        z = encoder(x)
        decoder(x)
    else:
        break
