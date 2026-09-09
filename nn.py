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

    return nn, torch


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


if __name__ == "__main__":
    app.run()
