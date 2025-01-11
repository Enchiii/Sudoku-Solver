import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class InvertColors:
    def __call__(self, image):
        return 255 - image


class RemoveAlphaChannel:
    def __call__(self, img):
        if img.shape[-1] == 4:
            rgb = img[..., :3]
            alpha = img[..., 3] / 255.0
            background = np.array((255, 255, 255), dtype=np.float32)
            blended = (rgb * alpha[..., None] + background * (1 - alpha[..., None]))
            return blended.astype(np.uint8)

        else:
            return img


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = nn.Dropout2d()(x)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = nn.Dropout()(x)
        x = self.fc2(x)
        return F.softmax(x)
