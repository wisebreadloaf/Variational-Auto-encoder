import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn
from cvae import ConvAutoencoder
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20

NUM_EPOCHS = 120
BATCH_SIZE = 10
LR_RATE = 1e-4

dataset = datasets.MNIST(
    root="dataset/", train=True, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

loop = enumerate(train_loader)
count = 0
for i, (x, y) in loop:
    if count < 1:
        print(x.shape)
        print(y)
        for i in range(int(y.shape)):
            save_image(x[i].unsqueeze(), f"./try/generated_{y[i]}.png")
        count += 1
    else:
        break
