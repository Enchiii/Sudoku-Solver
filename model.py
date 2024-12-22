import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import numpy as np
from sklearn.datasets import load_digits
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 11)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


batch_size = 4
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '#')

digits = load_digits()
data = digits.images
labels = digits.target
data = np.expand_dims(data, axis=1)

transform = transforms.Compose([
    lambda x: torch.tensor(x, dtype=torch.float32),  # Convert NumPy array to tensor
    transforms.Resize((16, 16))
])

# Apply transform to dataset
data = torch.stack([transform(img) for img in data])  # Convert each image
labels = torch.tensor(labels, dtype=torch.long)

# Determine how many black images to add for the '#' class
n_black_images = len(labels) // 10  # Add ~10% of the dataset size
noise_probability = 0.01  # Probability of a pixel being turned into noise

# Create a black image and repeat it
black_image = torch.zeros((1, 16, 16), dtype=torch.float32)  # Single black image
black_images = black_image.repeat(n_black_images, 1, 1, 1)  # Repeat black image

# Add random small noise
noise = torch.rand_like(black_images) < noise_probability  # Create a mask for noise
black_images[noise] = 1.0  # Add noise to black images

black_labels = torch.full((n_black_images,), 10, dtype=torch.long)  # Class label `10`

# Combine original and new data
data = torch.cat([data, black_images], dim=0)
labels = torch.cat([labels, black_labels], dim=0)

# Wrap into a TensorDataset
dataset = TensorDataset(data, labels)

# Split into training and testing datasets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])

# DataLoader configurations
batch_size = 4
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

if __name__ == '__main__':
    for epoch in range(5):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, _data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = _data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    print('Finished Training')

    PATH = './models/m1.pth'
    torch.save(net.state_dict(), PATH)

    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # print images
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

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
