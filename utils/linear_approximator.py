import numpy as np

class linapx:
    """A linear approximator that minimize least squire errors.
    """
    def __init__(self, x, fx, add_bias=False) -> None:
        """Given dataset (x, fx), fit a linear function of x to approximate fx.

        Args:
            x (np.ndarray): Nxn
            fx (np.ndarray): Nxd, value of f(x)
            contain_bias (bool, optional): Indication whether x contains bias coefficient, 
                i.e. a column of ones as the first column. Defaults to False.
        """
        self.x = x
        self.fx = fx

        if add_bias:
            X = np.column_stack((np.ones(self.x.shape[0]), self.x))
        self.coefficient, self.residuals, self.rank, self.s = np.linalg.lstsq(X, self.fx, rcond=None)

    def predict(self, X=None, add_bias=False):
        """Given a set of data points X, predict the value of f(X)

        Args:
            X (np.ndarray, optional): Mxn. Defaults to None
            contain_bias (bool, optional): Indication whether X contains bias coefficient, 
                i.e. a column of ones as the first column. Defaults to False.

        Returns:
            y_hat (np.ndarray): Mxd. Predicted values of f(X). 
        """

        if X is None:
            X = self.x

        if add_bias:
            X = np.column_stack((np.ones(X.shape[0]), X))
            y_hat = X @ self.coefficient
        return y_hat

