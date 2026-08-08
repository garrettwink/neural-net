import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    import marimo as mo

    return mo, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Tensor - 3+ dimensional matrix
    """)
    return


@app.cell
def _(torch):
    # torch.tensor
    scalar = torch.tensor(7)
    print(scalar.type(), scalar)
    # still of type torch.tensor
    return (scalar,)


@app.cell
def _(scalar):
    scalar.ndim
    # returns dimensions of scalar
    return


@app.cell
def _(scalar):
    scalar.item()
    #retrieve contents of tensor
    return


@app.cell
def _(torch):
    vector = torch.tensor([1,2,3])
    print(vector, vector.ndim, vector.shape)
    return


@app.cell
def _(torch):
    matrix = torch.tensor([[1,2],
                           [3,4]])
    print(matrix, matrix.ndim)
    return (matrix,)


@app.cell
def _(matrix):
    matrix.shape
    return


@app.cell
def _(torch):
    tensor = torch.tensor([[[1,2,3],
                [4,5,6],
                [7,8,9]]])

    tensor.ndim
    tensor.shape
    return


@app.cell
def _(torch):
    tensor2 = torch.tensor([[[1,2],[3,4],[5,6]]])
    tensor2.shape
    return


@app.cell
def _(torch):
    # random
    random_tensor = torch.rand(size=(3,4))
    random_tensor, random_tensor.dtype
    return


@app.cell
def _(torch):
    image = torch.rand(size=(512,512,3))
    image.shape
    return


@app.cell
def _(torch):
    #zeroes and ones
    zeros = torch.zeros(size=(4,4))
    zeros
    return


@app.cell
def _(torch):
    # arange
    range = torch.arange(start=0, end=100, step=4)
    range
    return (range,)


@app.cell
def _(range, torch):
    range0 = torch.zeros_like(range)
    range0
    return


@app.cell
def _(torch):
    tensor_cmds = torch.tensor([2,3],
                               dtype=torch.float64,
                               device=None,
                                requires_grad=False)
    tensor_cmds
    return


if __name__ == "__main__":
    app.run()
