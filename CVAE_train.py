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
BATCH_SIZE = 256
LR_RATE = 1e-4

dataset = datasets.MNIST(
    root="dataset/", train=True, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)


model = ConvAutoencoder().to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr=LR_RATE)
loss_fn = nn.BCELoss(reduction="sum")

for epoch in range(NUM_EPOCHS):
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (x, _) in loop:
        x = x.to(DEVICE)
        x_rec, mu, sigma = model(x)

        rec_loss = loss_fn(x_rec, x)
        kl_div = -0.5 * torch.sum(
            1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2)
        )

        loss = rec_loss + kl_div
        optim.zero_grad()
        loss.backward()
        optim.step()

        loop.set_postfix(loss=loss.item())

torch.save(model.state_dict(), "cvae_model.pth")
model = model.to("cuda")


def inference(digit, num_examples=1):
    images = []
    idx = 0
    for x, y in dataset:
        if y == digit:
            images.append(x.to(DEVICE))
            idx += 1
        if idx == 10:
            break

    encodings_digit = []
    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encode(images[d].unsqueeze(0))
        encodings_digit.append((mu, sigma))

    mu, sigma = encodings_digit[digit]
    for example in range(num_examples):
        print(
            f"example generation started: ./outputs/generated_{digit}_ex{example}.png"
        )
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        out = model.decode(z)
        out = out.view(-1, 1, 28, 28)
        save_image(out, f"./outputs/CVAEgenerated_{digit}_ex{example}.png")


for idx in range(10):
    inference(idx, num_examples=5)
