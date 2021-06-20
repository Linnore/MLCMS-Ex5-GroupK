import numpy as np

def lst_1d(x, fx):
    """This method return least square results for predicting fx from x, where x is one-dimensional data.

    Args:
        x (float): Nx1 numpy array
        fx (float): Nx1 numpy array, true value of f(x)

    Return:
        The results of least squire estimate, in the format same as numpy.linalg.lstsq()
    """
    X = np.column_stack((np.ones(x.shape[0]), x)) # Attached the coefficient for bias
    return np.linalg.lstsq(X, fx, rcond=None)


def lst_radial_basis(x, fx, e=1, L=10, randomseed=20):

    """Given dataset D having two columns (x, fx) and N rows, return the least square results for the problem || fx - phi(x) @ C.T ||, 
    where phi(x) = (phi_1(x), phi_2(x), ..., phi_L(x)), c_i is the coeffient for each phi_i and C = (c_1, c_2, ..., c_L)^T. 
    The radial basis is used here, i.e. phi_i(x) = exp(-r/e^3) where r=||x_i-x|| and x_i randomly chosen in the dataset

    Args:
        x (float): Nx1 numpy narray
        fx (float): Nx1 numpy narray, true value of f(x)
        e (float): the bandwidth in the radial basis
        L (int): dimension of the radial basis
        randomseed(int): the seed for randomly chosen center x_i for the basis function
    """
    
    # Randomly choose the center for basis function
    N = x.shape[0]
    np.random.seed(randomseed)
    center = x[np.random.choice(np.arange(0,N), size=L, replace=False)].copy()

    # Get phi(x)
    px = np.zeros((N, L))
    for i, xi in enumerate(x):
        r = (center-xi).reshape(-1,1)
        r = np.linalg.norm(r, axis=1)
        #print(i, r)

        px[i] = np.exp(-r**2 / e**2)
    

    return px, np.linalg.lstsq(px, fx, rcond=None)






    return 

