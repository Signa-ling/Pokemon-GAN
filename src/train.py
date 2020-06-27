import os

from utils import dataload, trainer


def main():
    # 設定
    modes = ['DCGAN', 'SAGAN']
    mode = modes[1]
    image_path = './images/'
    size = (64, 64)
    batch_size = 8
    epochs = 300
    z_dim = 100
    g_lr, d_lr = 0.0001, 0.0004
    betas = [0.5, 0.999]
    # RGBA → RGBにするならコメントアウトを消す
    convert_flag = True
    if convert_flag:
        channel = 3
    else:
        channel = 4

    weight_path = f'./Weight/{mode}/'
    generate_path = f'./Generated_Image/{mode}/'

    if not os.path.exists(weight_path):
        os.makedirs(f'{weight_path}Discriminator/')
        os.makedirs(f'{weight_path}Generator/')

    if not os.path.exists(generate_path):
        os.makedirs(generate_path)

    paths = [weight_path, generate_path]

    data_loader = dataload.get_data_loader(image_path, size,
                                           batch_size, convert_flag)
    train = trainer.Trainer(channel, data_loader, epochs, batch_size)
    if mode == 'DCGAN':
        train.train_dcgan(g_lr, d_lr, betas, paths, z_dim)
    elif mode == 'SAGAN':
        train.train_sagan(g_lr, d_lr, betas, paths, z_dim)

    print("Finish")


if __name__ == "__main__":
    main()
