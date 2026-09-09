import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    from torch.utils.data import Dataset
    from torchvision.transforms import v2
    import matplotlib.pyplot as plt

    return DataLoader, datasets, nn, plt, torch, v2


@app.cell
def _(torch):
    dev = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    return (dev,)


@app.cell
def _(nn):
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    return (NeuralNetwork,)


@app.cell
def _(NeuralNetwork, dev):
    model = NeuralNetwork().to(dev)
    print(model)
    return (model,)


@app.cell
def _(dev, model, nn, torch):
    X = torch.rand(1, 28, 28, device=dev)
    logits = model(X)
    preds = nn.Softmax(dim=1)(logits)
    pred = preds.argmax(1)
    return


@app.cell
def _(nn, torch):
    input_image = torch.rand(3,28,28)
    flatten = nn.Flatten()
    flat_image = flatten(input_image)
    print(flat_image.size())
    return (flat_image,)


@app.cell
def _(flat_image, nn):
    layer1 = nn.Linear(28*28, 20)
    hidden1 = layer1(flat_image)
    print(hidden1.size())
    return


@app.cell
def _(torch):


    x = torch.ones(5)  # input tensor
    y = torch.zeros(3)  # expected output
    w = torch.randn(5, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)
    z = torch.matmul(x, w)+b
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
    print(f"Gradient function for z = {z.grad_fn}")
    print(f"Gradient function for loss = {loss.grad_fn}")
    return b, loss, w


@app.cell
def _(b, loss, w):
    loss.backward()
    print(w.grad)
    print(b.grad)
    return


@app.cell
def _(model, nn, torch):
    learning_rate = 1e-3
    batch_size = 64
    epochs = 5
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    return batch_size, epochs, loss_fn, optimizer


@app.cell
def _(batch_size, torch):
    def train_loop(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X,y) in enumerate(dataloader):
            pred = model(X)
            loss = loss_fn(pred, y)

            #backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * batch_size + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    def test_loop(dataloader, model, loss_fn):
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X,y in dataloader:
                pred = model(X)
                test_loss += loss_fn(pred,y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    

    return test_loop, train_loop


@app.cell
def _(DataLoader, datasets, plt, torch, v2):
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        download=False
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        download=False
    )

    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }

    figure = plt.figure(figsize=(8,8))
    cols, rows = 3,3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    return test_dataloader, train_dataloader


@app.cell
def _(
    epochs,
    loss_fn,
    model,
    optimizer,
    test_dataloader,
    test_loop,
    train_dataloader,
    train_loop,
):

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")
    return


if __name__ == "__main__":
    app.run()
