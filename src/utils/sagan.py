# Todo: Generate, Discriminatorの実装

import torch
import torch.nn as nn


class SelfAttention(nn.Module):

    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()

        self.query_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=in_dim//8,
                                    kernel_size=1)

        self.key_conv = nn.Conv2d(in_channels=in_dim,
                                  out_channels=in_dim//8,
                                  kernel_size=1)

        self.value_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=in_dim,
                                    kernel_size=1)

        self.softmax = nn.Softmax(dim=-2)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        X = x
        proj_query = self.query_conv(X).view(
            X.shape[0], -1, X.shape[2]*X.shape[3])
        proj_query = proj_query.permute(0, 2, 1)
        proj_key = self.key_conv(X).view(
            X.shape[0], -1, X.shape[2]*X.shape[3])
        S = torch.bmm(proj_query, proj_key)

        attention_map_T = self.softmax(S)
        attention_map = attention_map_T.permute(0, 2, 1)

        proj_value = self.value_conv(X).view(
            X.shape[0], -1, X.shape[2]*X.shape[3])

        o = torch.bmm(proj_value, attention_map.permute(0, 2, 1))

        o = o.view(X.shape[0], X.shape[1], X.shape[2], X.shape[3])
        out = x + self.gamma * o

        return out, attention_map


class Generator(nn.Module):

    def __init__(self, z_dim=100, image_size=64, channel=4):
        super(Generator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.utils.spectral_norm(
                nn.ConvTranspose2d(
                    z_dim, image_size * 4, 4, 1, 0, bias=False)),
            nn.BatchNorm2d(image_size * 4),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.utils.spectral_norm(
                nn.ConvTranspose2d(image_size*4, image_size*2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(image_size*2),
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.utils.spectral_norm(
                nn.ConvTranspose2d(image_size*2, image_size, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(image_size),
            nn.ReLU(inplace=True)
        )

        self.self_attention1 = SelfAttention(in_dim=image_size)

        self.layer4 = nn.Sequential(
            nn.utils.spectral_norm(
                nn.ConvTranspose2d(image_size, image_size//2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(image_size//2),
            nn.ReLU(inplace=True)
        )

        self.self_attention2 = SelfAttention(in_dim=image_size//2)

        self.last = nn.Sequential(
            nn.ConvTranspose2d(image_size//2, channel, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        layers = [self.layer1, self.layer2, self.layer3,
                  self.self_attention1, self.layer4,
                  self.self_attention2, self.last]
        attention_maps = []

        for i, layer in enumerate(layers):
            if i == 3 or i == 5:
                z, attention_map = layer(z)
                attention_maps.append(attention_map)
            else:
                z = layer(z)
                # print(i, z.shape)  # 形状確認

        return z, attention_maps


class Descriminator(nn.Module):

    def __init__(self, z_dim=100, image_size=64, channel=4):
        super(Descriminator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(channel, image_size//2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(image_size//2, image_size, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(image_size),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(image_size, image_size*2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(image_size*2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.self_attention1 = SelfAttention(in_dim=image_size*2)

        self.layer4 = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(image_size*2, image_size*4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(image_size*4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.self_attention2 = SelfAttention(in_dim=image_size*4)

        self.last = nn.Conv2d(image_size*4, 1, 4, 1, 0, bias=False)

    def forward(self, x):
        layers = [self.layer1, self.layer2, self.layer3,
                  self.self_attention1, self.layer4,
                  self.self_attention2, self.last]
        attention_maps = []

        for i, layer in enumerate(layers):
            if i == 3 or i == 5:
                x, attention_map = layer(x)
                attention_maps.append(attention_map)
            else:
                x = layer(x)
                # print(i, x.shape)  # 形状確認

        return x, attention_maps
