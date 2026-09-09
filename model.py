import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import torch
    import matplotlib.pyplot as plt

    return mo, plt, torch


@app.cell
def _(torch):
    X = torch.arange(start=0, end=1, step = 0.01)
    b = 0.7
    m = 2
    y = m*X + b
    X[:10], y[:10]
    return X, y


@app.cell
def _(X, y):
    train_split = int(0.8*len(X))
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]

    len(X_train), len(y_train), len(X_test), len(y_test)
    return X_test, X_train, y_test, y_train


@app.cell
def _(X_test, X_train, plt, y_test, y_train):
    def plot_predictions(train_data=X_train, 
                         train_labels=y_train, 
                         test_data=X_test, 
                         test_labels=y_test, 
                         predictions=None):
      """
      Plots training data, test data and compares predictions.
      """
      plt.figure(figsize=(10, 7))

      # Plot training data in blue
      plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
  
      # Plot test data in green
      plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

      if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

      # Show the legend
      plt.legend(prop={"size": 14})

    return (plot_predictions,)


@app.cell
def _(plot_predictions, plt):
    plot_predictions()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Building Model
    """)
    return


@app.cell
def _(torch):
    from torch import nn
    class LinearRegressionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = nn.Parameter(torch.randn(1,
                                                    requires_grad=True,
                                                    dtype=torch.float))

            self.bias = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float))

            # forward method
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.weights * x + self.bias

    return (LinearRegressionModel,)


@app.cell
def _(LinearRegressionModel):
    model_0 = LinearRegressionModel()
    list(model_0.parameters())

    return (model_0,)


@app.cell
def _(model_0):
    model_0.state_dict()
    return


@app.cell
<<<<<<< HEAD
def _(torch):
    shape = (2,3)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)

    print(f"Random Tensor: \n {rand_tensor} \n")
    print(f"Ones Tensor: \n {ones_tensor} \n")
    print(f"Zeros Tensor: \n {zeros_tensor}")
    return


@app.cell
def _(torch):
    ten = torch.tensor([[1,2],[3,4]])
=======
def _(X_test, model_0, torch, y_test):
    with torch.inference_mode():
        y_preds = model_0(X_test)

    y_preds, y_test
    return (y_preds,)


@app.cell
def _(plot_predictions, plt, y_preds):
    plot_predictions(predictions=y_preds)
    plt.gca() 
>>>>>>> 8b8cdf08a232b0cda27e99cb140a3656f2816dfe
    return


if __name__ == "__main__":
    app.run()
