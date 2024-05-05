from captcha.image import ImageCaptcha
from process import image_to_tensor
import pandas as pd
import string
import torch
from torch.utils.data import Dataset
#import random


def create_images(path='captchas/', num_images=100):
    image = ImageCaptcha(width=100, height=100)
    for i in range(num_images):
        for digit in string.digits:
            image.write(digit, path + f'{digit}/{i}.png')


def create_dataset(path='captchas/',out='dataset/dataset.hdf', num_images=100):
    data = {'image': [], 'label': []}
    for i in range(num_images):
        for digit in string.digits:
            data['image'].append(image_to_tensor(path + f'{digit}/{i}.png').numpy())
            data['label'].append(int(digit))
    df = pd.DataFrame(data)
    df.to_hdf(out, key='data', mode='w')


if __name__ == '__main__':
    create_images(num_images=1000)
    create_dataset(num_images=1000)