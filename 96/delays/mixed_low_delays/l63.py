"""
Module for simulation of the Lorenz '63 model.
"""

import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def l63_tendency(X, sigma, rho, beta):
    """
    Calculate tendencies for the Lorenz '63 model.

    Args:
        X (ndarray): (n inits, 3) Current states
        sigma (float): Prandtl number
        rho (float): Rayleigh
        beta (float): Some spatial scale

    Returns:
        dXdt (ndarray): (n inits, 3) Array of tendencies
    """

    dXdt = np.zeros_like(X)
    dXdt[..., 0] = sigma * (X[..., 1] - X[..., 0])
    dXdt[..., 1] = X[..., 0] * (rho - X[..., 2]) - X[..., 1]
    dXdt[..., 2] = X[..., 0] * X[..., 1] - beta * X[..., 2]
    return dXdt


@jit(nopython=True, cache=True)
def integrate_l63(X0, dt, n_steps, sigma=10., rho=28., beta=8/3):
    """
    Integrate the Lorenz '63 model with RK4.

    Args:
        X0 (ndarray): (n_inits, 3) Initial state values
        dt (float): Size of the integration time step in MTU
        n_steps (int): Number of time steps integrated forward
        sigma (float): Prandtl number
        rho (float): Rayleigh
        beta (float): Some spatial scale

    Returns:
        X_series: (n_inits, n_timesteps, 3) Time series of states
    """

    X_series = np.zeros((X0.shape[0], n_steps, X0.shape[1]))
    X = X0.copy()
    X_series[:, 0] = X0
    k1_dXdt = np.zeros(X.shape)
    k2_dXdt = np.zeros(X.shape)
    k3_dXdt = np.zeros(X.shape)
    k4_dXdt = np.zeros(X.shape)

    for n in range(1, n_steps):
        k1_dXdt[:] = l63_tendency(X, sigma, rho, beta)
        k2_dXdt[:] = l63_tendency(X + k1_dXdt * dt / 2, sigma, rho, beta)
        k3_dXdt[:] = l63_tendency(X + k2_dXdt * dt / 2, sigma, rho, beta)
        k4_dXdt[:] = l63_tendency(X + k3_dXdt * dt, sigma, rho, beta)
        X += (k1_dXdt + 2 * k2_dXdt + 2 * k3_dXdt + k4_dXdt) / 6 * dt

        X_series[:, n] = X
    return X_series
