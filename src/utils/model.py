import torch
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, z_dim=100, image_size=64, channel=4):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(z_dim, image_size * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(image_size * 4),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(image_size*4, image_size*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size*2),
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(image_size*2, image_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(image_size, image_size//2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size//2),
            nn.ReLU(inplace=True)
        )

        self.last = nn.Sequential(
            nn.ConvTranspose2d(image_size//2, channel, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        layers = [self.layer1, self.layer2, self.layer3, self.layer4,
                  self.last]
        for i, layer in enumerate(layers):
            z = layer(z)
            # print(i, z.shape)  # 形状確認

        return z


class Descriminator(nn.Module):

    def __init__(self, z_dim=100, image_size=64, channel=4):
        super(Descriminator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(channel, image_size//2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(image_size//2, image_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(image_size, image_size*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size*2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(image_size*2, image_size*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size*4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.last = nn.Conv2d(image_size*4, 1, 4, 1, 0, bias=False)

    def forward(self, x):
        layers = [self.layer1, self.layer2, self.layer3, self.layer4,
                  self.last]
        for i, layer in enumerate(layers):
            x = layer(x)
            # print(i, x.shape)  # 形状確認

        return x
