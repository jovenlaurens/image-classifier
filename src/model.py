
import torch
import math

from torch import nn
from tqdm import tqdm
from torchvision import transforms

import matplotlib.pyplot as plt


class Model(nn.Module):
    def __init__(self):
        """_summary_
        Conv2d(x) = ((x - kernel_size + 2 * padding) / Stride) + 1 ) 
        MaxPool2d(x) = ((x + 2 * padding - dilation kernel_size) / Stride) + 1
        ConvTranspose2d(x) = (x - 1) * Stride - 2 * padding + kernel_size + output_padding
        """
        super(Model, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=3, padding=1),  # b, 64, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 64, 5, 5
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # b, 64, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),  # b, 64, 2, 2
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=2),  # b, 64, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 5, stride=3,
                               padding=1),  # b, 64, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        # Apply the transform to each image in the batch
        # print('encoder size: ', x.shape)
        x = self.decoder(x)
        # print('decoder size: ', x.shape)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def extract_features(self, x):
        x = self.encoder(x)
        x = x.flatten(start_dim=1)
        return x

    def summary(self):
        print(self)


if __name__ == '__main__':
    print(Model())
