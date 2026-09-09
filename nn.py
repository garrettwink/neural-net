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

    return (torch,)


@app.cell
def _(torch):
    dev = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    return


app._unparsable_cell(
    r"""
    class NeuralNetwork(nn.Module):
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
