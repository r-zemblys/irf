"""
@author: Raimondas Zemblys
@email: raimondas.zemblys@gmail.com
"""
import os
import time
import copy
import itertools
import re
from distutils.dir_util import mkpath

import numpy as np
import scipy.signal as sg
import scipy.interpolate as interp
from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage.morphology import binary_dilation
import scipy.io as scio

# import astropy.stats as ast

from util_lib.utils import round_up_to_odd, round_up, rolling_window, vcorrcoef
from util_lib.etdata import aggr_events, get_px2deg

ft_all = [
    "fs",
    "disp",
    "vel",
    "acc",
    "mean-diff",
    "med-diff",
    "rms",
    "std",
    "bcea",
    "rms-diff",
    "std-diff",
    "bcea-diff",
    "rayleightest",
    "i2mc",
]


def get_i2mc(etdata, fpath_i2mc, geom):
    """Load i2mc features
    Parameters:
        etdata      --  instance of ETData()
        fpath_i2mc  --  path for loading/saving i2mc features
        geom        --  dictionary with following parameters of setup geometry:
                    screen_width
                    screen_height
                    eye_distance
                    display_width_pix
                    display_height_pix
    Returns:
        i2mc features if exists. Saves mat files for using it with i2mc extractor
    """
    if os.path.exists(fpath_i2mc):
        i2mc = scio.loadmat(fpath_i2mc)
    else:
        i2mc = None
        # convert to mat formal
        data = etdata.data
        px2deg = get_px2deg(geom)

        t = (data["t"] - data["t"].min()) * 1000
        t = np.float64(t[:, np.newaxis])

        X = np.float64(data["x"][:, np.newaxis] * px2deg + geom["display_width_pix"] / 2)
        Y = np.float64(data["y"][:, np.newaxis] * px2deg + geom["display_height_pix"] / 2)
        X[np.isnan(X)] = -geom["display_width_pix"]
        Y[np.isnan(Y)] = -geom["display_height_pix"]

        data_mat = dict({"time": t, "right": dict({"X": X, "Y": Y}), "fs": etdata.fs})

        fdir, fname = os.path.split(fpath_i2mc)
        mkpath(fdir)
        scio.savemat("{}/{}_raw{}".format(fdir, *os.path.splitext(fname)), data_mat)
        print("%s does not exist. Run i2mc extractor first!" % fpath_i2mc)
    return i2mc


def calc_fix_pos(etdata, fix, w=50):
    """
    TODO: dublicate function. Update to use calc_event_data
    """
    data = etdata.data
    ws = round_up_to_odd(w / 1000.0 * etdata.fs)
    fix_pos = []
    for f in fix:
        ind_s = f[0] + ws
        ind_s = ind_s if ind_s < f[1] else f[1]
        ind_e = f[1] - ws
        ind_e = ind_e if ind_e > f[0] else f[0]

        posx_s = np.nanmean(data[f[0] : ind_s]["x"])
        posy_s = np.nanmean(data[f[0] : ind_s]["y"])
        posx_e = np.nanmean(data[ind_e : f[1]]["x"])
        posy_e = np.nanmean(data[ind_e : f[1]]["y"])
        fix_pos.append([posx_s, posx_e, posy_s, posy_e])
    return np.array(fix_pos)


def extract_features(etdata, **kwargs):
    """Extracts features for IRF"""
    tic = time.time()

    # get parameters
    data = etdata.data
    w, w_vel, w_dir = kwargs["w"], kwargs["w_vel"], kwargs["w_dir"]

    # find sampling rate
    fs = etdata.fs

    # window size for spatial measures in samples
    ws = round_up_to_odd(w / 1000.0 * fs + 1)

    # window size in samples for velocity calculation
    ws_vel = round_up_to_odd(w_vel / 1000.0 * fs)

    # window size in samples for direction calculation
    ws_dir = round_up_to_odd(w_dir / 1000.0 * fs)

    mask_interp = np.zeros(len(data), dtype=np.bool)
    """Legacy code. Interpolates through missing points.
    if "interp" in kwargs and kwargs["interp"]:
        r = np.arange(len(data))
        _mask = np.isnan(data["x"]) | np.isnan(data["y"])
        fx = interp.PchipInterpolator(r[~_mask], data[~_mask]["x"], extrapolate=True)
        fy = interp.PchipInterpolator(r[~_mask], data[~_mask]["y"], extrapolate=True)
        data["x"][_mask] = fx(r[_mask])
        data["y"][_mask] = fy(r[_mask])
        mask_interp = _mask
    """

    # prepare data for vectorized processing
    roll = (ws - 1) // 2
    ws_pad = (max((ws, ws_vel, ws_dir)) - 1) // 2

    x_padded = np.pad(data["x"], (ws_pad, ws_pad), "constant", constant_values=np.nan)
    y_padded = np.pad(data["y"], (ws_pad, ws_pad), "constant", constant_values=np.nan)

    ws_dir_pad = (ws_dir - 1) // 2
    x_padded_dir = np.pad(data["x"], (ws_dir_pad, ws_dir_pad), "constant", constant_values=np.nan)
    y_padded_dir = np.pad(data["y"], (ws_dir_pad, ws_dir_pad), "constant", constant_values=np.nan)

    x_windowed = rolling_window(x_padded, ws)
    y_windowed = rolling_window(y_padded, ws)
    dx_windowed = rolling_window(np.diff(x_padded), ws - 1)
    dy_windowed = rolling_window(np.diff(y_padded), ws - 1)
    x_windowed_dir = rolling_window(np.diff(x_padded_dir), ws_dir - 1)
    y_windowed_dir = rolling_window(np.diff(y_padded_dir), ws_dir - 1)

    # Extract features
    features = {"fs": np.ones(len(data)) * fs}

    for d, dd in zip(["x", "y"], [x_windowed, y_windowed]):
        # difference between positions of preceding and succeding windows,
        # aka tobii feature, together with data quality features and its variants
        means = np.nanmean(dd, axis=1)
        meds = np.nanmedian(dd, axis=1)
        features[f"mean-diff-{d}"] = np.roll(means, -roll) - np.roll(means, roll)
        features[f"med-diff-{d}"] = np.roll(meds, -roll) - np.roll(meds, roll)

        # standard deviation
        features[f"std-{d}"] = np.nanstd(dd, axis=1)
        features[f"std-next-{d}"] = np.roll(features["std-%s" % d], -roll)
        features[f"std-prev-{d}"] = np.roll(features["std-%s" % d], roll)

    features["mean-diff"] = np.hypot(features["mean-diff-x"], features["mean-diff-y"])
    features["med-diff"] = np.hypot(features["med-diff-x"], features["med-diff-y"])

    features["std"] = np.hypot(features["std-x"], features["std-y"])
    features["std-diff"] = np.hypot(features["std-next-x"], features["std-next-y"]) - np.hypot(
        features["std-prev-x"], features["std-prev-y"]
    )

    # BCEA
    P = 0.68  # cumulative probability of area under the multivariate normal
    k = np.log(1 / (1 - P))
    # rho = [np.corrcoef(px, py)[0,1] for px, py in zip(x_windowed, y_windowed)]
    rho = vcorrcoef(x_windowed, y_windowed)
    features["bcea"] = (
        2 * k * np.pi * features["std-x"] * features["std-y"] * np.sqrt(1 - np.power(rho, 2))
    )
    features["bcea-diff"] = np.roll(features["bcea"], -roll) - np.roll(features["bcea"], roll)

    # RMS
    features["rms"] = np.hypot(
        np.sqrt(np.nanmean(np.square(dx_windowed), axis=1)),
        np.sqrt(np.nanmean(np.square(dy_windowed), axis=1)),
    )
    features["rms-diff"] = np.roll(features["rms"], -roll) - np.roll(features["rms"], roll)

    # disp, aka idt feature
    x_range = np.nanmax(x_windowed, axis=1) - np.nanmin(x_windowed, axis=1)
    y_range = np.nanmax(y_windowed, axis=1) - np.nanmin(y_windowed, axis=1)
    features["disp"] = x_range + y_range

    # velocity and acceleration
    features["vel"] = (
        np.hypot(
            sg.savgol_filter(data["x"], ws_vel, 2, 1), sg.savgol_filter(data["y"], ws_vel, 2, 1)
        )
        * fs
    )

    features["acc"] = (
        np.hypot(
            sg.savgol_filter(data["x"], ws_vel, 2, 2), sg.savgol_filter(data["y"], ws_vel, 2, 2)
        )
        * fs**2
    )

    # rayleightest
    # angl = np.arctan2(y_windowed_dir, x_windowed_dir)
    # features["rayleightest"] = ast.rayleightest(angl, axis=1)
    features["rayleightest"] = np.zeros(len(data))

    # i2mc
    if "i2mc" in kwargs and kwargs["i2mc"] is not None:
        features["i2mc"] = kwargs["i2mc"]["finalweights"].flatten()
    else:
        features["i2mc"] = np.zeros(len(data))

    # remove padding and nans
    mask_nans = np.any([np.isnan(values) for key, values in features.items()], axis=0)
    mask_pad = np.zeros_like(data["x"], dtype=np.bool)
    mask_pad[:ws_pad] = True
    mask_pad[-ws_pad:] = True
    mask = mask_nans | mask_pad | mask_interp
    features = {key: values[~mask].astype(np.float32) for key, values in features.items()}

    dtype = np.dtype([*zip(features.keys(), itertools.repeat(np.float32))])
    features = np.core.records.fromarrays(features.values(), dtype=dtype)

    # return features
    toc = time.time()
    if "print_et" in kwargs and kwargs["print_et"]:
        print(f"Feature extraction took {(toc - tic)}:.3f s.")
    return features, ~mask


def post_process(etdata, pred, pred_mask, events=(1, 2, 3), dev=False, **kwargs):
    status = pred_mask
    fs = etdata.fs

    # prepare array for storing event data
    events_pp = np.zeros((len(etdata.data), len(events) + 1))
    events_pp[pred_mask, 1:] = pred

    # filter raw probabilities
    events_pp = gaussian_filter1d(events_pp, 1, axis=0)
    events_pp[~pred_mask, 0] = 1

    #    #1. mark short interpolation sequences as valid,
    #    #i.e. remove short interpolation (or other "invalid data") events
    #    thres_id_s = round_up_to_odd(kwargs['thres_id']*fs/1000.+1)
    #    status_aggr = np.array(aggr_events(pred_mask))
    #    events_interp = status_aggr[status_aggr[:,-1]==False]
    #    md = events_interp[:,1] - events_interp[:,0]
    #    mask_rem_interp = md<thres_id_s
    #    ind_rem_interp=[i for s, e in events_interp[mask_rem_interp, :2]
    #                      for i in range(s, e)]
    #    status[ind_rem_interp] = True
    #
    #    ind_leave_interp=[i for s, e in events_interp[~mask_rem_interp, :2]
    #                        for i in range(s, e)]
    #    events_pp[ind_leave_interp, 0]=1
    #    events_pp[ind_leave_interp, 1:]=-1

    # 2. merge fixations; can be implemented as hpp
    thres_ifa = kwargs["thres_ifa"]
    thres_ifi = kwargs["thres_ifi"]
    thres_ifi_s = round_up(thres_ifi * fs / 1000.0)

    _events = np.argmax(np.around(events_pp, 3), axis=1)
    _events_aggr = np.array(aggr_events(_events))
    _events_fix = _events_aggr[_events_aggr[:, -1] == 1]
    _fix_pos = calc_fix_pos(etdata, _events_fix)

    # inter-fixation amplitudes
    ifa = np.pad(
        np.hypot(_fix_pos[:-1, 1] - _fix_pos[1:, 0], _fix_pos[:-1, 3] - _fix_pos[1:, 2]),
        (0, 1),
        "constant",
        constant_values=thres_ifa + 1e-5,
    )
    # inter-fixation intervals
    ifi = np.pad(
        _events_fix[1:, 0] - _events_fix[:-1, 1],
        (0, 1),
        "constant",
        constant_values=thres_ifi_s + 1,
    )
    mask_merge_fix = (ifa < thres_ifa) & (ifi < thres_ifi_s)

    ind_merge_fix = [
        i
        for s, e in zip(_events_fix[mask_merge_fix, 1], _events_fix[np.roll(mask_merge_fix, 1), 0])
        for i in range(s, e)
    ]
    events_pp[ind_merge_fix, 0] = -1
    events_pp[ind_merge_fix, 1] = 1
    events_pp[ind_merge_fix, 2:] = -1

    # 3.1 expand saccades; can be implemented as hpp
    thres_sd_s = kwargs["thres_sd_s"]  # make saccades to be at least 3 samples
    _events = np.argmax(np.around(events_pp, 3), axis=1)
    _events_aggr = np.array(aggr_events(_events))
    _events_sacc = _events_aggr[_events_aggr[:, -1] == 2]
    _sd = _events_sacc[:, 1] - _events_sacc[:, 0]
    mask_expand_sacc = _sd < thres_sd_s
    ind_mid_sacc = (
        _events_sacc[mask_expand_sacc][:, 1] - _events_sacc[mask_expand_sacc][:, 0]
    ) // 2 + _events_sacc[mask_expand_sacc][:, 0]
    ind_rem_fix = [
        i
        for s, e in zip(
            ind_mid_sacc - (thres_sd_s // 2 + thres_sd_s % 2), ind_mid_sacc + (thres_sd_s // 2)
        )
        for i in range(s, e)
    ]
    events_pp[ind_rem_fix, :2] = -1
    # events_pp[ind_rem_fix, 3] = -1
    events_pp[ind_rem_fix, 2] = 1

    # 3.2 merge nearby saccades; can be implemented as hpp
    thres_isi = kwargs["thres_isi"]
    thres_isi_s = round_up(thres_isi * fs / 1000.0)
    _events = np.argmax(np.around(events_pp, 3), axis=1)
    _events_aggr = np.array(aggr_events(_events))
    _events_sacc = _events_aggr[_events_aggr[:, -1] == 2]
    # inter-saccade intervals
    isi = np.pad(
        _events_sacc[1:, 0] - _events_sacc[:-1, 1],
        (0, 1),
        "constant",
        constant_values=thres_isi_s + 1,
    )
    mask_merge_sacc = isi < thres_isi_s
    ind_merge_sacc = [
        i
        for s, e in zip(
            _events_sacc[mask_merge_sacc, 1], _events_sacc[np.roll(mask_merge_sacc, 1), 0]
        )
        for i in range(s, e)
    ]
    events_pp[ind_merge_sacc, 2] = 1
    events_pp[ind_merge_sacc, :2] = -1

    # 3.3. remove too short or too long saccades.
    # + for too long
    # - for too short; give a chance to become fixation samples
    # leave too long for hpp
    thres_sd_lo = kwargs["thres_sd_lo"]
    #    thres_sd_hi=kwargs['thres_sd_hi']
    thres_sd_lo_s = round_up(thres_sd_lo * fs / 1000.0)
    #    thres_sd_hi_s = round_up(thres_sd_hi*fs/1000.)
    _events = np.argmax(np.around(events_pp, 3), axis=1)
    _events_aggr = np.array(aggr_events(_events))
    _events_sacc = _events_aggr[_events_aggr[:, -1] == 2]
    fd = _events_sacc[:, 1] - _events_sacc[:, 0]
    mask_rem_sacc = fd < thres_sd_lo_s  # | (fd>thres_sd_hi_s)
    ind_rem_sacc = [i for s, e in _events_sacc[mask_rem_sacc, :2] for i in range(s, e)]
    events_pp[ind_rem_sacc, 2:] = -1

    # 4. remove unreasonable PSOs; give a chance to become other classes
    _events = np.argmax(np.around(events_pp, 3), axis=1)
    _events_aggr = np.array(aggr_events(_events))
    mask_pso = _events_aggr[:, -1] == 3

    # remove too short PSOs; not used
    #    thres_pd = kwargs['thres_pd']
    #    thres_pd_s = round_up(thres_pd*fs/1000.)
    #    pso_dur = _events_aggr[:,1]-_events_aggr[:,0]
    #    mask_pso_dur = pso_dur < thres_pd_s

    # remove PSOs not after saccade
    seq = "".join(map(str, _events_aggr[:, -1]))
    mask_pso_after_sacc = np.ones_like(mask_pso)
    pso_after_sacc = [_m.start() + 1 for _m in re.finditer("(?=23)", seq)]
    mask_pso_after_sacc[pso_after_sacc] = False

    # mask_inv_pso = mask_pso & mask_pso_dur & mask_pso_after_sacc
    mask_inv_pso = mask_pso & mask_pso_after_sacc

    ind_inv_pso = [i for s, e in _events_aggr[mask_inv_pso, :2] for i in range(s, e)]
    events_pp[ind_inv_pso, 2:] = -1  # can't be neither pso, neither saccade

    # 5. remove too short fixations; +
    # leave for hpp
    #    thres_fd=kwargs['thres_fd']
    #    thres_fd_s = round_up(thres_fd*fs/1000.)
    #    _events = np.argmax(np.around(events_pp, 3), axis=1)
    #    _events_aggr = np.array(aggr_events(_events))
    #    _events_fix = _events_aggr[_events_aggr[:,-1]==1]
    #    fd = _events_fix[:,1]-_events_fix[:,0]
    #    mask_rem_fix=fd<thres_fd_s
    #    ind_rem_fix=[i for s, e in _events_fix[mask_rem_fix, :2]
    #                            for i in range(s, e)]
    #    events_pp[ind_rem_fix, 0]=1
    #    events_pp[ind_rem_fix, 1:]=-1

    """legacy code
    #6.1 blink detection: remove saccade-like events between missing data
    #leave for hpp
    _events = np.argmax(np.around(events_pp, 3), axis=1)
    _events_aggr = np.array(aggr_events(_events))

    seq = ''.join(map(str, _events_aggr[:,-1]))
    patterns = ['20', '02'] # !!! only works for patterns ['20', '02'] !!!
    _blinks = [_m.start() for _pattern in patterns
                         for _m in re.finditer('(?=%s)'%_pattern, seq) ]
    if len(_blinks):
        _blinks = np.array(_blinks)
        _blinks = np.unique(np.concatenate([_blinks, _blinks+1]))
        ind_blink=[i for s, e in _events_aggr[_blinks, :2]
                              for i in range(s, e)]
        events_pp[ind_blink, 0]=1
        events_pp[ind_blink, 1:]=-1

    #6.2 remove PSOs again, because some of saccades might been removed; +
    #leave for hpp
    _events = np.argmax(np.around(events_pp, 3), axis=1)
    _events_aggr = np.array(aggr_events(_events))
    mask_pso = _events_aggr[:,-1]==3
    seq = ''.join(map(str, _events_aggr[:,-1]))
    mask_pso_after_sacc = np.ones_like(mask_pso)
    pso_after_sacc = [_m.start()+1 for _m in re.finditer('(?=23)', seq) ]
    mask_pso_after_sacc[pso_after_sacc]=False
    mask_inv_pso = mask_pso & mask_pso_after_sacc
    ind_inv_pso=[i for s, e in _events_aggr[mask_inv_pso, :2]
                            for i in range(s, e)]

    events_pp[ind_inv_pso, 1:]=-1 #remove event completely
    """

    # 7. Final events
    events = np.argmax(np.around(events_pp, 3), axis=1)
    status = ~(events == 0)
    return events, status, events_pp


class hpp:
    """Implements hard post-processing of event data"""

    def __init__(self):

        self.check_accum = {
            "short_saccades": 0,
            "long_saccades": 0,
            "saccades": 0,
            "sacc202": 0,  # saccades surrouding undef
            "sacc20": 0,  # saccade before undef
            "sacc02": 0,  # saccade after undef
            "sacc_isi": 0,
            "short_fixations": 0,
            "fixations": 0,
            "short_pso": 0,
            "pso": 0,
            "pso23": 0,  # proper pso
            "pso13": 0,  # pso after fixation
            "pso03": 0,  # pso after undefined
        }

        self.check_inds = {
            "short_saccades": 0,
            "long_saccades": 0,
            "saccades": 0,
            "sacc202": 0,  # saccades surrouding undef
            "sacc20": 0,  # saccade before undef
            "sacc02": 0,  # saccade after undef
            "sacc_isi": 0,
            "short_fixations": 0,
            "fixations": 0,
            "short_pso": 0,
            "pso": 0,
            "pso23": 0,  # proper pso
            "pso13": 0,  # pso after fixation
            "pso03": 0,  # pso after undefined
        }

    def reset_accum(self):
        """Resets check accumulator"""
        for k, v in self.check_accum.items():
            self.check_accum[k] = 0

    def run_pp(self, etdata, pp=True, **kwargs):
        """Performs post-processing sanity check and hard-removes or replaces events
        Parameters:
            etdata  --  instance of ETData()
            pp      --  if True, post-processing is performed. Otherwise only counts cases
        Returns:
            Numpy array of updated event stream
            Numpy array of updated status
            Dictionary with event status accumulator
        """

        _evt = etdata.calc_evt(fast=True)
        status = etdata.data["status"]
        fs = etdata.fs

        check = self.check_accum

        # pp
        # Saccade sanity check
        _sacc = _evt.query("evt==2")
        check["saccades"] += len(_sacc)

        # check isi
        _isi = (_sacc[1:]["s"].values - _sacc[:-1]["e"].values) / fs
        _isi_inds = np.where(_isi < kwargs["thres_isi"] / 1000.0)[0]
        check["sacc_isi"] += len(_isi_inds)
        self.check_inds["sacc_isi"] = _sacc.index[_isi_inds].values

        # TODO: implement isi merging
        #        if pp:
        #
        #            _etdata = copy.deepcopy(etdata)
        #            _evt_unfold = [_e for _, e in _evt.iterrows()
        #                          for _e in itertools.repeat(e['evt'],
        #                                                     int(np.diff(e[['s', 'e']])))]
        #
        #        _etdata.data['evt'] = _evt_unfold
        #        _etdata.calc_evt(fast=True)
        #        _evt = _etdata.evt

        # pp: remove short saccades
        #        _sdur_thres = max([0.006, float(3/etdata.fs)])
        #        _sdur = _evt.query('evt==2 and dur<@_sdur_thres')

        thres_sd_lo = kwargs["thres_sd_lo"] / 1000.0
        thres_sd_lo_s = round_up(thres_sd_lo * fs)

        _sdur = _evt.query("evt==2 and (dur<@thres_sd_lo or dur_s<@thres_sd_lo_s)")
        check["short_saccades"] += len(_sdur)
        self.check_inds["short_saccades"] = _sdur.index.values
        if pp:
            _evt.loc[_sdur.index, "evt"] = 0

        # check long saccades.
        thres_sd_hi = kwargs["thres_sd_hi"] / 1000.0
        thres_sd_hi_s = round_up(thres_sd_hi * fs)
        _sdur = _evt.query("evt==2 and dur_s>@thres_sd_hi_s")
        check["long_saccades"] += len(_sdur)
        self.check_inds["long_saccades"] = _sdur.index.values
        if pp:
            _evt.loc[_sdur.index, "evt"] = 0

        # pp: find saccades surrounding undef;
        _sacc_check = {"202": 0, "20": 0, "02": 0}
        seq = "".join(map(str, _evt["evt"]))
        for pattern in _sacc_check.keys():
            _check = np.array([m.start() for m in re.finditer("(?=%s)" % pattern, seq)])
            if not (len(_check)):
                continue

            _sacc_check[pattern] += len(_check)
            self.check_inds["sacc%s" % pattern] = _check
        #            #pp: remove saccades surrounding undef; not used anymore
        #            if pp:
        #                if (pattern=='202'):
        #                    assert ((_evt.loc[_check+1, 'evt']==0).all() and
        #                            (_evt.loc[_check+2, 'evt']==2).all())
        #                    _evt.loc[_check, 'evt'] = 0
        #                    _evt.loc[_check+2, 'evt'] = 0
        #
        ##                if (pattern=='20'):
        ##                    assert (_evt.loc[_check+1, 'evt']==0).all()
        ##                    _evt.loc[_check, 'evt'] = 0
        ##                if (pattern=='02'):
        ##                    assert (_evt.loc[_check+1, 'evt']==2).all()
        ##                    _evt.loc[_check+1, 'evt'] = 0
        #                seq=''.join(map(str, _evt['evt']))

        check["sacc202"] += _sacc_check["202"]
        check["sacc20"] += _sacc_check["20"]
        check["sacc02"] += _sacc_check["02"]

        # PSO sanity check
        check["pso"] += len(_evt.query("evt==3"))

        # pp: change short PSOs to fixations; not used
        #        thres_pd = kwargs['thres_pd']/1000.
        #        thres_pd_s = round_up(thres_pd*fs)
        #        _pdur = _evt.query('evt==3 and (dur<@thres_pd or dur_s<@thres_pd_s)')
        #        check['short_pso']+=len(_pdur)
        #        self.check_inds['short_pso'] = _pdur.index.values
        #        if pp:
        #            _evt.loc[_pdur.index, 'evt'] = 1

        # pp: remove unreasonable psos
        _pso_check = {"13": 0, "03": 0, "23": 0}
        seq = "".join(map(str, _evt["evt"]))
        for pattern in _pso_check.keys():
            _check = np.array([m.start() for m in re.finditer("(?=%s)" % pattern, seq)])
            if not (len(_check)):
                continue

            _pso_check[pattern] += len(_check)
            self.check_inds["pso%s" % pattern] = _check
            # pp: change PSOs after fixations to fixations
            if pp:
                if pattern == "13":
                    assert (_evt.loc[_check + 1, "evt"] == 3).all()
                    _evt.loc[_check + 1, "evt"] = 1
                # pp: change PSOs after undef to undef
                if pattern == "03":
                    assert (_evt.loc[_check + 1, "evt"] == 3).all()
                    _evt.loc[_check + 1, "evt"] = 0
        check["pso23"] += _pso_check["23"]
        check["pso13"] += _pso_check["13"]
        check["pso03"] += _pso_check["03"]

        # fixation sanity check
        # unfold and recalculate event data
        _evt_unfold = [
            _e
            for _, e in _evt.iterrows()
            for _e in itertools.repeat(e["evt"], int(np.diff(e[["s", "e"]])))
        ]
        _etdata = copy.deepcopy(etdata)
        _etdata.data["evt"] = _evt_unfold
        _etdata.calc_evt(fast=True)
        _evt = _etdata.evt

        check["fixations"] += len(_evt.query("evt==1"))

        # pp: remove short fixations
        thres_fd = kwargs["thres_fd"] / 1000.0
        thres_fd_s = round_up(thres_fd * fs)
        _fdur = _evt.query("evt==1 and (dur<@thres_fd or dur_s<@thres_fd_s)")
        check["short_fixations"] += len(_fdur)
        self.check_inds["short_fixations"] = _fdur.index.values

        # TODO:
        # check fixation merge
        if pp:
            _inds = np.array(_fdur.index)
            _evt.loc[_inds, "evt"] = 0
        #            #check if there are saccades or psos left around newly taged undef
        #            #so basically +- 2 events around small fixation
        #            _inds = np.unique(np.concatenate((_inds, _inds+1, _inds-1, _inds+2, _inds-2)))
        #            _inds = _inds[(_inds>-1) & (_inds<len(_evt))]
        #            _mask =_evt.loc[_inds, 'evt'].isin([2, 3])
        #            _evt.loc[_inds[_mask.values], 'evt'] = 0

        # return result
        _evt_unfold = [
            _e
            for _, e in _evt.iterrows()
            for _e in itertools.repeat(e["evt"], int(np.diff(e[["s", "e"]])))
        ]
        assert len(_evt_unfold) == len(status)

        status[np.array(_evt_unfold) == 0] = False

        return np.array(_evt_unfold), status, check, self.check_inds
