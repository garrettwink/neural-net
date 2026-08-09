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


if __name__ == "__main__":
    app.run()
