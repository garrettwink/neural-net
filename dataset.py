import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    from torch.utils.data import Dataset
    from torchvision import datasets
    from torchvision.transforms import v2
    import matplotlib.pyplot as plt

    return datasets, torch, v2


@app.cell
def _(datasets, torch, v2):
    train_data = datasets.FashionMNIST(
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

    return


if __name__ == "__main__":
    app.run()
