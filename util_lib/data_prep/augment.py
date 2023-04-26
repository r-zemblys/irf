"""
@author: Raimondas Zemblys
@email: raimondas.zemblys@gmail.com
"""
##
import os
import glob
import copy
import random
from distutils.dir_util import mkpath

import parse
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.signal as sg

try:
    from __init__ import ROOT as IRF_ROOT
except ImportError:
    IRF_ROOT = "./"

from util_lib.etdata import ETData
from util_lib import utils

plt.rcParams["image.cmap"] = "gray"
plt.rc("axes.spines", top=False, right=False)
plt.ioff()

if __name__ == "__main__":
    # setup directories
    ROOT = os.path.join(IRF_ROOT, "etdata")
    EXP = "lookAtPoint_EL"

    ROOT_OUTPUT = os.path.join(ROOT, EXP, "augment")
    ROOT_TRAIN = os.path.join(ROOT, EXP, "training")
    ROOT_TEST = os.path.join(ROOT, EXP, "testing")

    for _path in [
        ROOT_OUTPUT,
        ROOT_TRAIN,
        os.path.join(ROOT_TRAIN, "train"),
        os.path.join(ROOT_TRAIN, "val"),
        ROOT_TEST,
    ]:
        mkpath(_path)

    # select subjects
    subjects = [1, 2, 4, 5, 6]

    # adhere to the original split
    subjects_test = [4]
    subjects_train = [1, 2, 5, 6]

    # # random split
    # random.seed(0x062217)
    # random.shuffle(subjects)
    # subjects_test = subjects[:1]
    # subjects_train = subjects[1:]

    # setup other parameters
    sampling_rates = [1250, 1000, 500, 300, 250, 200, 120, 60, 30]
    lowpass_size = 20.0  # ms

    # setup noise mapping
    FWHM = 20  # approximate extent of data
    xrms = 3  # rms multiplier
    delta = 0.1  # resolution

    # TODO: this can be optimized
    N = 10  # noise levels
    rms_s = 0.005
    rms_levels = [0]
    for i in range(N):
        rms_levels.append(round(rms_s, 3))
        rms_e = rms_s * xrms
        rms_s = rms_e - rms_s

    # calculates sigma for full width at \part\ maximum
    w = np.hypot(FWHM, FWHM)
    sigma = w * 2 / (2.0 * np.sqrt(2 * np.log(xrms)))

    x = np.arange(-FWHM, FWHM, delta)
    y = np.arange(-FWHM, FWHM, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = utils.bivariate_normal(X, Y, sigma, sigma, 0.0, 0.0)

    s = np.ptp(Z1)
    m = Z1.min()

    # draw noise function
    # plt.ion()
    # rms=rms_levels[1]
    # Z1*=-1
    # Z1+=s+m
    # Z1*=(rms*(xrms-1)/s)
    # Z1+=rms
    #
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, Z1)

    ##
    # augment data
    etdata = ETData()
    np.random.seed(0x062217)

    for fpath in glob.glob(os.path.join(ROOT, EXP, "*.npy")):
        fdir, fname = os.path.split(os.path.splitext(fpath)[0])
        sub = parse.parse("lookAtPoint_EL_S{sub:d}", fname).named["sub"]

        # skip if subject is not included
        if not (sub in subjects):
            continue

        data = etdata.load(fpath)
        # interpolate to avoid artifacts when resampling
        mask_interp = np.isnan(data["x"]) | np.isnan(data["y"]) | (~data["status"])
        assert np.all(mask_interp == ~data["status"])

        for _d in ["x", "y"]:
            _f = interp1d(data["t"][~mask_interp], data[_d][~mask_interp])
            data[_d][mask_interp] = _f(data["t"][mask_interp])

        # resample
        _iter_fs = tqdm(sampling_rates)
        for fs in _iter_fs:
            _iter_fs.set_description(f"Subject: {sub}, sampling rate: {fs}")
            if fs == 1000:
                # do not filter original data
                _data_ds = copy.deepcopy(data)
            else:
                # low-pass filter
                _data = copy.deepcopy(data)
                nyq_rate = fs / 2.0
                cutoff_hz = 0.8 * nyq_rate
                numtaps = np.int16(fs * lowpass_size / 1000.0)
                if numtaps < 2:
                    numtaps = 2

                b, a = sg.butter(numtaps, cutoff_hz / nyq_rate)
                _data["x"] = sg.filtfilt(b, a, _data["x"])
                _data["y"] = sg.filtfilt(b, a, _data["y"])

                # resample
                t_interp = np.arange(_data["t"][0], _data["t"][-1], 1.0 / fs)
                # setup containers for resampled data
                if fs > 1000:
                    _data_ds = np.zeros_like(np.hstack((_data, _data)))[: len(t_interp)]
                else:
                    _data_ds = np.zeros_like(_data)[: len(t_interp)]
                _data_ds["t"] = t_interp

                for var, interp_type in zip(
                    ["x", "y", "status", "evt"],
                    ["slinear", "slinear", "nearest", "nearest"],
                ):
                    _f = interp1d(_data["t"], _data[var], kind=interp_type)
                    _data_ds[var] = _f(_data_ds["t"])

            # add noise
            for rms in rms_levels:
                _data_noise = copy.deepcopy(_data_ds)

                # iterate through samples and get noise level for each
                n = []
                for _sample in _data_noise:
                    X, Y = np.meshgrid(_sample["x"], _sample["y"])
                    v = utils.bivariate_normal(X, Y, sigma, sigma, 0.0, 0.0)[0][0]
                    v *= -1
                    v += s + m
                    v *= rms * (xrms - 1) / s
                    v += rms
                    n.append(v)

                noise_x, noise_y = utils.box_muller_gaussian(
                    np.random.uniform(0, 1, len(_data_noise)),
                    np.random.uniform(0, 1, len(_data_noise)),
                )
                n = np.array(n) / 2.0

                noise_x *= n
                noise_y *= n
                _data_noise["x"] += noise_x
                _data_noise["y"] += noise_y

                # remove interpolated samples
                _data_noise["x"][~_data_noise["status"]] = np.nan
                _data_noise["y"][~_data_noise["status"]] = np.nan

                # save data
                etdata.load(_data_noise, **{"source": "array"})
                etdata.save(os.path.join(ROOT_OUTPUT, f"{fname}_{fs}_{rms:.3f}"))

                # train/val split
                _len = len(_data_noise)
                val_ind = np.random.randint(0.25 * _len, 0.5 * _len)
                offset = val_ind + int(0.25 * _len)

                # save split data
                if sub in subjects_train:
                    _spath_train = os.path.join(ROOT_TRAIN, "train", f"{fname}_{fs}_{rms:.3f}")
                    _spath_val = os.path.join(ROOT_TRAIN, "val", f"{fname}_{fs}_{rms:.3f}")

                    np.save(f"{_spath_train}_train1", _data_noise[:val_ind])
                    np.save(f"{_spath_train}_train2", _data_noise[offset:])
                    np.save(f"{_spath_val}_val", _data_noise[val_ind:offset])
                else:
                    np.save(os.path.join(ROOT_TEST, f"{fname}_{fs}_{rms:.3f}_test"), _data_noise)
