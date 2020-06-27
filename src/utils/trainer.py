import time

import torch
from torch import nn, optim
from torchvision.utils import save_image

from utils import dcgan, sagan


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def _epoch_start(epoch, epochs):
    print('=' * 20)
    print(f'Epoch {epoch}/{epochs}')
    print('-' * 20)
    print('（train）')


def _epoch_result(epoch, epoch_time, loss_Ds, loss_Gs):
    print('-------------')
    print(f'epoch {epoch}')
    print(f'loss_D: {loss_Ds:.4f} || loss_G: {loss_Gs:.4f}')
    print(f'timer: {epoch_time:.4f}')


def _saves(model_G, model_D, epoch, check_z, weight_path, gen_path):
    torch.save(model_G.state_dict(),
               f'{weight_path}Generator/G_{epoch:03d}.prm',
               pickle_protocol=4)
    torch.save(model_D.state_dict(),
               f'{weight_path}Discriminator/D_{epoch:03d}.prm',
               pickle_protocol=4)
    gen_img, _ = model_G(check_z)
    print(type(gen_img))
    save_image(gen_img, f"{gen_path}{epoch:03d}.png")


class Trainer():
    def __init__(self, channel, data_loader, epochs, batch_size):
        self.channel = channel
        self.data_loader = data_loader
        self.epochs = epochs
        self.batch_size = batch_size

    def train_dcgan(self, g_lr, d_lr, betas, paths, z_dim=100):
        model_G = dcgan.Generator(channel=self.channel).to(device)
        model_D = dcgan.Descriminator(channel=self.channel).to(device)
        g_opt = optim.Adam(model_G.parameters(), g_lr, betas)
        d_opt = optim.Adam(model_D.parameters(), d_lr, betas)
        check_z = torch.randn(self.batch_size, z_dim, 1, 1).to(device)
        loss_f = nn.BCEWithLogitsLoss(reduction='mean')

        print(model_G)
        print('=' * 50)
        print(model_D)

        model_G.train()
        model_D.train()

        for epoch in range(self.epochs):
            time_epoch_start = time.time()
            _epoch_start(epoch, self.epochs)
            loss_Gs, loss_Ds = 0.0, 0.0

            for images in self.data_loader:
                if images.size()[0] == 1:
                    continue

                batch_len = len(images)
                images = images.to(device)
                # batch_len = images.size()[0]
                real_label = torch.ones(batch_len).to(device)
                fake_label = torch.zeros(batch_len).to(device)

                # Discriminatorの訓練
                # 真画像生成, loss算出
                real_out = model_D(images)
                loss_D_real = loss_f(real_out.view(-1), real_label)

                # 偽画像生成, loss算出
                z = torch.randn(batch_len, z_dim, 1, 1).to(device)
                fake_img = model_G(z)
                fake_out = model_D(fake_img)
                loss_D_fake = loss_f(fake_out.view(-1), fake_label[:batch_len])

                # 誤差計算
                loss_D = loss_D_real + loss_D_fake

                # バックプロパゲーション
                g_opt.zero_grad()
                d_opt.zero_grad()
                loss_D.backward()
                d_opt.step()

                # Generatorの訓練
                # 偽画像生成 → 判定, loss算出
                z = torch.randn(batch_len, z_dim, 1, 1).to(device)
                fake_img = model_G(z)
                fake_out = model_D(fake_img)
                loss_G = loss_f(fake_out.view(-1), real_label[:batch_len])

                # バックプロパゲーション
                g_opt.zero_grad()
                d_opt.zero_grad()
                loss_G.backward()
                g_opt.step()

                loss_Ds += loss_D.item()
                loss_Gs += loss_G.item()

            # result
            t_epoch_finish = time.time()
            epoch_time = t_epoch_finish - time_epoch_start
            _epoch_result(epoch, epoch_time, loss_Ds, loss_Gs)

            # 訓練途中のモデル・生成画像の保存
            if epoch % 10 == 0:
                _saves(model_G, model_D, epoch,
                       check_z, paths[0], paths[1])

    def train_sagan(self, g_lr, d_lr, betas, paths, z_dim=100):
        model_G = sagan.Generator(channel=self.channel).to(device)
        model_D = sagan.Descriminator(channel=self.channel).to(device)
        g_opt = optim.Adam(model_G.parameters(), g_lr, betas)
        d_opt = optim.Adam(model_D.parameters(), d_lr, betas)
        check_z = torch.randn(self.batch_size, z_dim, 1, 1).to(device)

        print(model_G)
        print('=' * 50)
        print(model_D)

        model_G.train()
        model_D.train()

        for epoch in range(self.epochs):
            time_epoch_start = time.time()
            _epoch_start(epoch, self.epochs)
            loss_Gs, loss_Ds = 0.0, 0.0

            for images in self.data_loader:
                if images.size()[0] == 1:
                    continue

                batch_len = len(images)
                images = images.to(device)

                # Discriminatorの訓練
                # 真画像生成, loss算出
                real_out, _ = model_D(images)
                loss_D_real = nn.ReLU()(1.0 - real_out).mean()

                # 偽画像生成, loss算出
                z = torch.randn(batch_len, z_dim, 1, 1).to(device)
                fake_img, _ = model_G(z)
                fake_out, _ = model_D(fake_img)
                loss_D_fake = nn.ReLU()(1.0 + fake_out).mean()

                # 誤差計算
                loss_D = loss_D_real + loss_D_fake

                # バックプロパゲーション
                g_opt.zero_grad()
                d_opt.zero_grad()
                loss_D.backward()
                d_opt.step()

                # Generatorの訓練
                # 偽画像生成 → 判定, loss算出
                z = torch.randn(batch_len, z_dim, 1, 1).to(device)
                fake_img, _ = model_G(z)
                fake_out, _ = model_D(fake_img)
                loss_G = - fake_out.mean()

                # バックプロパゲーション
                g_opt.zero_grad()
                d_opt.zero_grad()
                loss_G.backward()
                g_opt.step()

                loss_Ds += loss_D.item()
                loss_Gs += loss_G.item()

            # result
            t_epoch_finish = time.time()
            epoch_time = t_epoch_finish - time_epoch_start
            _epoch_result(epoch, epoch_time, loss_Ds, loss_Gs)
            # 訓練途中のモデル・生成画像の保存
            if epoch % 10 == 0:
                _saves(model_G, model_D, epoch,
                       check_z, paths[0], paths[1])
