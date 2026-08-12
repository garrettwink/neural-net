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
def _(nn, torch):
    # Create a Linear Regression model class
    class LinearRegressionModel(torch.nn.Module): # <- almost everything in PyTorch is a nn.Module (think of this as neural network lego blocks)
        def __init__(self):
            super().__init__() 
            self.weights = nn.Parameter(torch.randn(1, # <- start with random weights (this will get adjusted as the model learns)
                                                    dtype=torch.float), # <- PyTorch loves float32 by default
                                       requires_grad=True) # <- can we update this value with gradient descent?)

            self.bias = nn.Parameter(torch.randn(1, # <- start with random bias (this will get adjusted as the model learns)
                                                dtype=torch.float), # <- PyTorch loves float32 by default
                                    requires_grad=True) # <- can we update this value with gradient descent?))

        # Forward defines the computation in the model
        def forward(self, x: torch.Tensor) -> torch.Tensor: # <- "x" is the input data (e.g. training/testing features)
            return self.weights * x + self.bias # <- this is the linear regression formula (y = m*x + b)

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
