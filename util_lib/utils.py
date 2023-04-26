"""
@author: Raimondas Zemblys
@email: raimondas.zemblys@gmail.com
"""
import os
import numpy as np


def round_up_to_odd(f, min_val=3):
    """Rounds input value up to nearest odd number.
    Parameters:
        f       --  input value
        min_val --  minimum value to retun
    Returns:
        Rounded value
    """
    w = np.int32(np.ceil(f) // 2 * 2 + 1)
    w = min_val if w < min_val else w
    return w


def round_up(f, min_val=3):
    """Rounds input value up.
    Parameters:
        f       --  input value
        min_val --  minimum value to retun
    Returns:
        Rounded value
    """
    w = np.int32(np.ceil(f))
    w = min_val if w < min_val else w
    return w


def rolling_window(a, window):
    """Implements effective rolling window
    Parameters:
        a       --  1D numpy array
        window  --  window size
    Returns:
        2D Numpy array, where each row is values from sliding window
    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def vcorrcoef(x, y):
    """
    NumPy vectorized correlation coefficient
    adapted from:
    https://waterprogramming.wordpress.com/2014/06/13/numpy-vectorized-correlation-coefficient/
    """
    xm = x - np.reshape(np.nanmean(x, axis=1), (x.shape[0], 1))
    ym = y - np.reshape(np.nanmean(y, axis=1), (y.shape[0], 1))
    r_num = np.nansum(xm * ym, axis=1)
    r_den = np.sqrt(np.nansum(xm**2, axis=1) * np.nansum(ym**2, axis=1))
    return r_num / r_den


def split_path(fpath):
    """Strips file extension and splits file path into directory and file name
    Parameters:
        fpath   -- full file path
    Returns:
        fdir    -- file directory
        fname   -- fine name without extension
    """
    return os.path.split(os.path.splitext(fpath)[0])


def box_muller_gaussian(u1, u2):
    z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
    return z1, z2


def bivariate_normal(x, y, sigmax=1.0, sigmay=1.0, mux=0.0, muy=0.0, sigmaxy=0.0):
    """
    Bivariate Gaussian distribution for equal shape *x*, *y*.
    See `bivariate normal
    <http://mathworld.wolfram.com/BivariateNormalDistribution.html>`_
    at mathworld.
    """
    x_mu = x - mux
    y_mu = y - muy

    rho = sigmaxy / (sigmax * sigmay)
    z = (
        x_mu**2 / sigmax**2
        + y_mu**2 / sigmay**2
        - 2 * rho * x_mu * y_mu / (sigmax * sigmay)
    )
    denom = 2 * np.pi * sigmax * sigmay * np.sqrt(1 - rho**2)
    return np.exp(-z / (2 * (1 - rho**2))) / denom
