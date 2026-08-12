import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import torch
    import matplotlib as plt

    return (torch,)


@app.cell
def _(torch):
    x = torch.arange(start=0, end=1, step = 0.05)
    b = 0.7
    m = 2
    y = m*x + b
    x[:10], y[:10]
    return


if __name__ == "__main__":
    app.run()
