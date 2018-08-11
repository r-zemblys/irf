#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: rz
@email: r.zemblys@tf.su.lt
"""
#%% imports
import os
import numpy as np

#%% functions and constants

def round_up_to_odd(f, min_val = 3):
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

def round_up(f, min_val = 3):
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

def vcorrcoef(X,Y):
    '''
    NumPy vectorized correlation coefficient
    adapted from:
    https://waterprogramming.wordpress.com/2014/06/13/numpy-vectorized-correlation-coefficient/
    '''
    Xm = X - np.reshape(np.mean(X,axis=1),(X.shape[0],1))
    Ym = Y - np.reshape(np.mean(Y,axis=1),(Y.shape[0],1))
    r_num = np.sum(Xm*Ym,axis=1)
    r_den = np.sqrt(np.sum(Xm**2,axis=1) * np.sum(Ym**2, axis=1))
    return r_num / r_den

def split_path(fpath):
    '''Strips file extension and splits file path into directory and file name
    Parameters:
        fpath   -- full file path
    Returns:
        fdir    -- file directory
        fname   -- fine name without extension
    '''
    return os.path.split(os.path.splitext(fpath)[0])