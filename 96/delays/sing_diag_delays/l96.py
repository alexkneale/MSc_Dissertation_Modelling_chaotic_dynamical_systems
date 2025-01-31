"""
Module for simulation of the Lorenz '96 model.
"""


import numpy as np
from numba import jit


@jit(nopython=True, cache=True, parallel=True)
def l96_tendency(X, Y, h, F, b, c):
    """
    Calculate tendencies for the X and Y variables of the Lorenz '96 model.

    Args:
        X (1D ndarray): Values of X variables at the current time step
        Y (1D ndarray): Values of Y variables at the current time step
        h (float): Coupling constant
        F (float): Forcing term
        b (float): Spatial scale ratio
        c (float): Time scale ratio

    Returns:
        dXdt (1D ndarray): Array of X increments,
        dYdt (1D ndarray): Array of Y increments
    """

    K = X.size
    J = Y.size // K
    dXdt = np.zeros(X.shape)
    dYdt = np.zeros(Y.shape)
    for k in range(K):
        dXdt[k] = (- X[k - 1] * (X[k - 2] - X[(k + 1) % K]) - X[k] + F
                   - h * c / b * np.sum(Y[k * J: (k + 1) * J]))
    for j in range(J * K):
        dYdt[j] = (
            - c * b * Y[(j + 1) % (J * K)] * (Y[(j + 2) % (J * K)] - Y[j-1])
            - c * Y[j] + h * c / b * X[int(j / J)])
    return dXdt, dYdt


@jit(nopython=True, cache=True, parallel=True)
def l96_reduced_tendency(X, F):
    """
    Calculate tendencies for the X variables of the reduced Lorenz '96 model.

    Args:
        X (1D ndarray): Values of X variables at the current time step
        F (float): Forcing term

    Returns:
        dXdt (1D ndarray): Array of X increments
    """

    K = X.size
    dXdt = np.zeros(X.shape)
    for k in range(K):
        dXdt[k] = - X[k - 1] * (X[k - 2] - X[(k + 1) % K]) - X[k] + F
    return dXdt


@jit(nopython=True, cache=True, parallel=True)
def integrate_l96(X_0, Y_0, dt, n_steps, h, F, b, c):
    """
    Integrate the Lorenz '96 model with RK4.

    Args:
        X_0 (1D ndarray): Initial X values.
        Y_0 (1D ndarray): Initial Y values.
        dt (float): Size of the integration time step in MTU
        n_steps (int): Number of time steps integrated forward.
        h (float): Coupling constant.
        F (float): Forcing term.
        b (float): Spatial scale ratio
        c (float): Time scale ratio

    Returns:
        X_series [number of timesteps, X size]: X values at each time step,
        Y_series [number of timesteps, Y size]: Y values at each time step
    """

    X_series = np.zeros((n_steps, X_0.size))
    Y_series = np.zeros((n_steps, Y_0.size))
    X = np.zeros(X_0.shape)
    Y = np.zeros(Y_0.shape)
    X[:] = X_0
    Y[:] = Y_0
    X_series[0] = X_0
    Y_series[0] = Y_0
    k1_dXdt = np.zeros(X.shape)
    k2_dXdt = np.zeros(X.shape)
    k3_dXdt = np.zeros(X.shape)
    k4_dXdt = np.zeros(X.shape)
    k1_dYdt = np.zeros(Y.shape)
    k2_dYdt = np.zeros(Y.shape)
    k3_dYdt = np.zeros(Y.shape)
    k4_dYdt = np.zeros(Y.shape)

    for n in range(1, n_steps):
        k1_dXdt[:], k1_dYdt[:] = l96_tendency(X, Y, h, F, b, c)
        k2_dXdt[:], k2_dYdt[:] = l96_tendency(X + k1_dXdt * dt / 2,
                                              Y + k1_dYdt * dt / 2,
                                              h, F, b, c)
        k3_dXdt[:], k3_dYdt[:] = l96_tendency(X + k2_dXdt * dt / 2,
                                              Y + k2_dYdt * dt / 2,
                                              h, F, b, c)
        k4_dXdt[:], k4_dYdt[:] = l96_tendency(X + k3_dXdt * dt,
                                              Y + k3_dYdt * dt,
                                              h=h, F=F, b=b, c=c)
        X += (k1_dXdt + 2 * k2_dXdt + 2 * k3_dXdt + k4_dXdt) / 6 * dt
        Y += (k1_dYdt + 2 * k2_dYdt + 2 * k3_dYdt + k4_dYdt) / 6 * dt
        X_series[n] = X
        Y_series[n] = Y

    return X_series, Y_series

@jit(nopython=True, cache=True, parallel=True)
def integrate_reduced_l96(X_0, dt, n_steps, F):
    """
    Integrate the reduced Lorenz '96 model with RK4.

    Args:
        X_0 (1D ndarray): Initial X values.
        dt (float): Size of the integration time step in MTU
        n_steps (int): Number of time steps integrated forward.
        F (float): Forcing term.

    Returns:
        X_series [number of timesteps, X size]: X values at each time step
    """

    X_series = np.zeros((n_steps, X_0.size))
    X = np.zeros(X_0.shape)
    X[:] = X_0
    X_series[0] = X_0
    k1_dXdt = np.zeros(X.shape)
    k2_dXdt = np.zeros(X.shape)
    k3_dXdt = np.zeros(X.shape)
    k4_dXdt = np.zeros(X.shape)

    for n in range(1, n_steps):
        k1_dXdt[:] = l96_reduced_tendency(X, F)
        k2_dXdt[:] = l96_reduced_tendency(X + k1_dXdt * dt / 2, F)
        k3_dXdt[:] = l96_reduced_tendency(X + k2_dXdt * dt / 2, F)
        k4_dXdt[:] = l96_reduced_tendency(X + k3_dXdt * dt, F)
        X += (k1_dXdt + 2 * k2_dXdt + 2 * k3_dXdt + k4_dXdt) / 6 * dt

        X_series[n] = X
    return X_series
