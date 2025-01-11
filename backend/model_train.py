import cv2
import torch
import os
import numpy as np
import torchvision.transforms as transforms
import torchvision
import torch.optim as optim
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image
from model import Net, InvertColors, RemoveAlphaChannel


batch_size = 64
classes = (' ', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# PATH = './backend/dataset/train/'
PATH = './dataset/train/'


transform = transforms.Compose([
    RemoveAlphaChannel(),
    InvertColors(),
    transforms.ToTensor(),
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(28, scale=(0.7, 1.0)),
    # dodac rozciagniecie
    # transforms.RandomHorizontalFlip(), # add when fine-tuning
])

n_images = len([name for name in os.listdir(PATH + '/1')])
noise_probability = 0.01

if not os.path.isdir(PATH + '/0'):
    os.makedirs(PATH + '/0')

for i in range(n_images):
    image = torch.full((28, 28, 3), 1, dtype=torch.float32)

    noise = torch.rand_like(image) < noise_probability
    image[noise] = 0

    image_pil = Image.fromarray((image.numpy()*255).astype('uint8'))

    image_pil.save(f"{PATH}/0/{i}.png")
else:
    print("DONE")


def image_loader(path='.'):
    return np.array(Image.open(path))


dataset = torchvision.datasets.ImageFolder(PATH, transform=transform, loader=image_loader)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

print("DONE")

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

if __name__ == '__main__':
    for epoch in range(8):

        running_loss = 0.0
        for i, _data in enumerate(dataloader, 0):
            inputs, labels = _data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    print('Finished Training')

    PATH = './models/m2.pth'
    # PATH = './backend/models/m1.pth'
    torch.save(net.state_dict(), PATH)

    print("saved")

    dataiter = iter(dataloader)
    images, labels = next(dataiter)

    # print test_images
    grid_img = torchvision.utils.make_grid(images).permute(1, 2, 0)
    grid_img = grid_img * 0.5 + 0.5  # Assuming normalization with mean=0.5, std=0.5
    plt.imshow(grid_img.numpy())
    plt.show()
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    net = Net()
    net.load_state_dict(torch.load(PATH, weights_only=True))

    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                  for j in range(4)))

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)

            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
