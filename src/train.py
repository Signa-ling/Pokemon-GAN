import os
from statistics import mean

import torch
from torch import nn, optim
from torchvision.utils import save_image
from tqdm import tqdm

from utils import dataload, model

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_dcgan(model_G, model_D, data_loader, g_opt, d_opt, z_dim, batch_size):
    log_loss_G, log_loss_D = [], []
    ones = torch.ones(batch_size).to(device)
    zeros = torch.zeros(batch_size).to(device)
    loss_f = nn.BCEWithLogitsLoss(reduction='mean')

    for real_img in tqdm(data_loader):
        batch_len = len(real_img)

        # Generatorの訓練
        z = torch.randn(batch_len, z_dim, 1, 1).to(device)
        fake_img = model_G(z)
        fake_img_tensor = fake_img.detach()

        out = model_D(fake_img)
        loss_G = loss_f(out.view(-1), ones[:batch_len])
        log_loss_G.append(loss_G.item())

        model_D.zero_grad()
        model_G.zero_grad()
        loss_G.backward()
        g_opt.step()

        # Discriminatorの訓練
        real_img = real_img.to(device)
        real_out = model_D(real_img)
        loss_D_real = loss_f(real_out.view(-1), ones[:batch_len])

        fake_img = fake_img_tensor
        fake_out = model_D(fake_img)
        loss_D_fake = loss_f(fake_out.view(-1), zeros[:batch_len])

        loss_D = loss_D_real + loss_D_fake
        log_loss_D.append(loss_D.item())

        model_D.zero_grad()
        model_G.zero_grad()
        loss_D.backward()
        d_opt.step()

    return mean(log_loss_G), mean(log_loss_D)


def main():
    # 設定
    image_path = './images/'
    size = (64, 64)
    batch_size = 8
    epochs = 300
    z_dim = 100
    g_lr, d_lr = 0.0001, 0.0004
    beta1, beta2 = 0.5, 0.999
    # RGBA → RGBにするならコメントアウトを消す
    convert_flag = True
    if convert_flag:
        channel = 3
    else:
        channel = 4

    weight_path = './Weight/'
    generate_path = './Generated_Image/'

    if not os.path.exists(weight_path):
        os.makedirs(f'{weight_path}Discriminator/')
        os.makedirs(f'{weight_path}Generator/')

    if not os.path.exists(generate_path):
        os.makedirs(generate_path)

    data_loader = dataload.get_data_loader(image_path, size, batch_size, convert_flag)
    model_G = model.Generator(channel=channel).to(device)
    model_D = model.Descriminator(channel=channel).to(device)

    print(model_G)
    print('=' * 50)
    print(model_D)

    g_opt = optim.Adam(model_G.parameters(), g_lr, [beta1, beta2])
    d_opt = optim.Adam(model_D.parameters(), d_lr, [beta1, beta2])

    check_z = torch.randn(batch_size, z_dim, 1, 1).to(device)

    print("Setting OK")
    for epoch in range(epochs):
        train_dcgan(model_G, model_D, data_loader, g_opt, d_opt,
                    z_dim, batch_size)

        # 訓練途中のモデル・生成画像の保存
        if epoch % 10 == 0:
            torch.save(
                model_G.state_dict(),
                f"{weight_path}Generator/G_{epoch:03d}.prm",
                pickle_protocol=4)
            torch.save(
                model_D.state_dict(),
                f"{weight_path}Discriminator/D_{epoch:03d}.prm",
                pickle_protocol=4)

            generated_img = model_G(check_z)
            save_image(generated_img,
                       f"{generate_path}{epoch:03d}.png")

    print("Finish")


if __name__ == "__main__":
    main()
