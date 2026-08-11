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
def _(torch):
    ## Matrix Multiplication
    #2:19


    tensor1 = torch.tensor([[1,2,3],
                            [4,5,6]])
    tensor2 = torch.tensor([[3,4],
                            [5,6],
                            [5,6]])
    return tensor1, tensor2


@app.cell
def _(tensor1, tensor2):
    # matmul and transpose
    tensor1 @ tensor2
    tensor2.T
    return


@app.cell
def _():
    # test push
    return


if __name__ == "__main__":
    app.run()
