import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    import torchvision
    from torch.utils.data import Dataset, DataLoader
    from torchvision import datasets, transforms
    from torchvision.transforms import v2
    from torchvision.io import decode_image
    import os
    import pandas as pd

    return DataLoader, Dataset, decode_image, os, pd, torch, torchvision, v2


@app.cell
def _():
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("sidharkal/sports-image-classification")

    print("Path to dataset files:", path)
    return


@app.cell
def _(pd):
    from sklearn.model_selection import train_test_split

    full_df = pd.read_csv('./data/sports-image-classification/train.csv')
    train_df, test_df = train_test_split(full_df, test_size=0.2, stratify=full_df.iloc[:, 1], random_state=42)

    train_df.to_csv('./data/sports-image-classification/train_split.csv', index=False)
    test_df.to_csv('./data/sports-image-classification/test_split.csv', index=False)
    return


@app.cell
def _(DataLoader, Dataset, decode_image, os, pd, torch, torchvision, v2):
    #custom dataset
    class CustomImageDataset(Dataset):
        def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
            self.img_labels = pd.read_csv(annotations_file)
            self.img_dir = img_dir
            self.transform = transform
            self.target_transform = target_transform
            self.classes = sorted(self.img_labels.iloc[:, 1].unique())          # e.g. ['cricket', 'soccer', ...]
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}       # {'cricket': 0, 'soccer': 1, ...}

        def __len__(self):
            return len(self.img_labels)

        def __getitem__(self, idx):
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
            image = decode_image(img_path, mode=torchvision.io.ImageReadMode.RGB)
            label_name = self.img_labels.iloc[idx, 1]
            label = self.class_to_idx[label_name]     # convert string → int
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            return image, label

    batch_size = 4

    transform = v2.Compose([
        v2.Resize((224, 224)),   # pick a fixed size — see note below
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = CustomImageDataset(annotations_file='./data/sports-image-classification/train_split.csv',
                                   img_dir='./data/sports-image-classification/train',
                                   transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = CustomImageDataset(annotations_file='./data/sports-image-classification/test_split.csv',
                                 img_dir='./data/sports-image-classification/test',   # same folder — images haven't moved
                                 transform=transform)

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    classes = ('Badminton', 'Cricket', 'Tennis', 'Soccer', 'Swimming', 'Karate', 'Wrestling')
        
    return batch_size, classes, trainloader


@app.cell
def _(batch_size, torch, torchvision, trainloader):
    import matplotlib.pyplot as plt
    import numpy as np

    # functions to show an image


    def imshow(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        mean = torch.tensor(mean).view(3, 1, 1)
        std = torch.tensor(std).view(3, 1, 1)
        img = img * std + mean   # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{labels[j]:5s}' for j in range(batch_size)))
    return


@app.cell
def _(classes, torch):
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    class Net(nn.Module):
        def __init__(self, input_size=224):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)

            # compute flattened size dynamically based on actual input size
            with torch.no_grad():
                dummy = torch.zeros(1, 3, input_size, input_size)
                dummy = self.pool(self.conv1(dummy))
                dummy = self.pool(self.conv2(dummy))
                flat_size = dummy.numel()

            self.fc1 = nn.Linear(flat_size, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, len(classes))   # num_classes = however many sports categories you have

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    return criterion, net, optimizer


@app.cell
def _(criterion, net, optimizer, trainloader):
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs1, labels1 = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs1)
            loss = criterion(outputs, labels1)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
