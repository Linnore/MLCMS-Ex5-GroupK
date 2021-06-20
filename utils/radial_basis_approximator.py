import numpy as np

class rbapx:
    """A nonlinear approximator of f(x) using radial basis.
    """

    def __init__(self, x, fx):
        self.x = x
        self.fx = fx
        self.L = None
        self.e = None
        self.center = None
        self.phix = None
        self.coefficient=None

    def fit(self, L, e=2**0.5, randomseed=118010142):
        """Fit an approximation using radial basis

        Args:
            L (int): Number of Gaussian kernels used
            e (float, optional): Bandwidth for the radial basis. Defaults to 2**0.5.
            randomseed (int, optional): Seed for randomly chosen centers for radial basis. Defaults to 118010142.

        """
        self.L = L
        self.e = e

        # Randomly choose the center for basis function
        x = self.x
        N = x.shape[0]
        np.random.seed(randomseed)
        self.center = x[np.random.choice(np.arange(0,N), size=L, replace=False)].copy()

        self.phix = px = self.getPhi(x, L, e)
        self.coefficient, self.residuals, self.rank, self.s = np.linalg.lstsq(px, self.fx, rcond=None)

    def getPhi(self, x, L, e=2**0.5):
        """Generate values of phi(x) given configuration of phi

        Args:
            x (np.ndarray): the data points of x
            L (int): number of radial basis
            e (double, optional): bandwidth for gaussian kernel. Defaults to 2**0.5.

        Returns:
            px (np.ndarray): NxL, \phi(x)
        """
        N = x.shape[0]
        px = np.zeros((N, L))
        for i, xi in enumerate(x):
            r = (self.center-xi).reshape(-1,1)
            r = np.linalg.norm(r, axis=1)
            px[i] = np.exp(-r**2 / e**2)
        return px

    def predict(self, x=None):
        if x is None:
            x = self.x
        px = self.getPhi(x, self.L, self.e)
        return px @ self.coefficient


