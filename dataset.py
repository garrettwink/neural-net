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

    return datasets, plt, torch, v2


@app.cell
def _(datasets, torch, v2):
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

    return test_data, training_data


@app.cell
def _(plt, torch, training_data):
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
    plt.show()
    return


@app.cell
def _(test_data, training_data):
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    return


if __name__ == "__main__":
    app.run()
