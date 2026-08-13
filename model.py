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


app._unparsable_cell(
    r"""
    class LinearRegressionModel(nn.Module):
    
    """,
    name="_"
)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
