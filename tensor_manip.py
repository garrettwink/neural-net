import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import torch

    return (torch,)


@app.cell
def _(torch):
    tensor = torch.tensor([1,2,3])
    print(tensor + 10)
    print(tensor * 10)
    return


@app.cell
def _():
    ## Matrix Multiplication
    #2:19

    return


@app.cell
def _(torch):
    # test push
    tensor1 = torch.tensor([[1,2,3],
                           [4,5,6]])
    tensor2 = torch.tensor([[5,6],
                           [6,5],
                           [8,3]])

    return (tensor1,)


@app.cell
def _(tensor1):
    tensor1.min()
    return


@app.cell
def _(torch):
    torch.cuda.is_available()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
