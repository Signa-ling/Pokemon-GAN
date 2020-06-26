import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CreateDatasets(Dataset):
    def __init__(self, dir_name, data_transform, convert_flag):
        self.dir_name = dir_name
        self.file = os.listdir(dir_name)
        self.data_transform = data_transform
        self.convert_flag = convert_flag

    def __len__(self):
        return len(self.file)

    def __getitem__(self, i):
        image = Image.open(self.dir_name + self.file[i])
        if self.convert_flag:
            image = image.convert('RGB')
        image = self.data_transform(image)
        return image


def data_argument(images_path):
    flip_transform = transforms.RandomHorizontalFlip(p=1)
    images = os.listdir(images_path)
    for image in images:
        flip_img = flip_transform(Image.open(images_path + image))
        flip_img.save(f'{images_path}flip_{image}')
    print(len(os.listdir(images_path)))


def get_data_loader(images_path, size, batch_size, convert_flag=False):
    transform = transforms.Compose([transforms.Resize(size),
                                    transforms.ToTensor()])
    # data_argument(images_path)
    dataset = CreateDatasets(images_path, transform, convert_flag)
    len(dataset)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader
