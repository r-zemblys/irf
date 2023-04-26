"""
@author: Raimondas Zemblys
@email: raimondas.zemblys@gmail.com
"""
import os
import sys
import glob
import argparse
import multiprocessing
import json
from datetime import datetime
from distutils.dir_util import mkpath

import parse
from tqdm import tqdm
import joblib

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score

from util_lib.etdata import ETData
from util_lib.utils import split_path
from util_lib.irf import extract_features, ft_all, get_i2mc


def get_arguments():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(
        description="Eye-movement event detection " "using Random Forest."
    )
    parser.add_argument("root", type=str, help="The path containing eye-movement data.")
    parser.add_argument("dataset", type=str, help="The directory containing experiment data.")
    parser.add_argument(
        "--output_dir", type=str, default=None, help="The directory to save output."
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of workers to use.")

    return parser.parse_args()


def _extract_features(root, dataset):
    print("Running feature extraction...")
    # Extract features for train and val
    i2mc_ok = True
    etdata = ETData()

    files = sorted(glob.glob(os.path.join(root, dataset, "*[train][val]*/*.npy")))
    for fpath in tqdm(files):
        fdir, fname = split_path(fpath)
        odir = fdir.replace(dataset, f"{dataset}.cache")

        # create output dirs
        odir_feat = os.path.join(odir, "feat")
        odir_evt = os.path.join(odir, "evt")
        mkpath(odir_feat)
        mkpath(odir_evt)

        spath_feat = os.path.join(odir_feat, f"feat_{fname}.npy")
        spath_evt = os.path.join(odir_evt, f"evt_{fname}.npy")

        # check if feature files already exist
        if not os.path.exists(spath_feat) or not os.path.exists(spath_evt):
            etdata.load(fpath)

            # remove other events
            evt_mask = np.in1d(etdata.data["evt"], config["events"])
            etdata.data["x"][~evt_mask] = np.nan
            etdata.data["y"][~evt_mask] = np.nan
            etdata.data["status"][~evt_mask] = False

            # extract features
            if "i2mc" in config["features"]:
                fpath_i2mc = os.path.join(odir, f"{fname}_i2mc.mat")
                i2mc = get_i2mc(etdata, fpath_i2mc, config["geom"])
                if i2mc is None:
                    i2mc_ok = False
                    continue
                else:
                    config["extr_kwargs"]["i2mc"] = i2mc

            irf_features, pred_mask = extract_features(etdata, **config["extr_kwargs"])

            # select required features
            x = irf_features[ft_all]
            x = np.array(x.tolist())
            y = etdata.data["evt"][pred_mask]

            # check for other events
            assert np.all(np.in1d(np.unique(y), config["events"]))
            # check lengths
            assert (len(x) == len(y)) and (len(x) == pred_mask.sum())

            # save
            np.save(spath_feat, x)
            np.save(spath_evt, y)

    if not i2mc_ok:
        sys.exit("I2MC feature extraction failed.")


def train(data_path, feature_mask):
    # TODO: handle different ordering
    x = []
    y = []
    files = sorted(glob.glob(os.path.join(data_path, "train/evt/*.npy")))
    for fpath in tqdm(files):
        path_feat = fpath.replace("evt", "feat")

        _x = np.load(path_feat)[:, feature_mask]
        _y = np.load(fpath)
        _y[_y == 3] = 2  # relabel PSO samples to be saccade samples
        x.append(_x)
        y.append(_y)

    x = np.concatenate(x)
    y = np.concatenate(y)

    # train IRF
    clf = RandomForestClassifier(
        n_estimators=config["n_trees"],
        max_depth=None,
        class_weight="balanced_subsample",
        max_features=3,
        n_jobs=n_jobs,
        verbose=3,
    )

    clf.fit(x, y)

    # save
    print("Saving model...")
    spath = os.path.join("models", f"irf_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    mkpath(spath)
    joblib.dump([config["features"], clf], os.path.join(spath, "model.pkl"), compress=9, protocol=2)
    print("...done")

    return clf, spath


def evaluate(clf, data_path, mpath, feature_mask):
    print("Running evaluation...")
    clf.set_params(verbose=0)
    fmt = "evt_lookAtPoint_EL_S{sub:d}_{fs:d}_{rms:.4f}{}"

    result = []
    files = sorted(glob.glob(os.path.join(data_path, "val/evt/*.npy")))
    for fpath in tqdm(files):
        fdir, fname = split_path(fpath)
        path_feat = fpath.replace("evt", "feat")

        # load data
        _x = np.load(path_feat)[:, feature_mask]
        _y = np.load(fpath)

        # predict
        pred_val = clf.predict_proba(_x)
        pred_val_class = np.argmax(pred_val, axis=1) + 1

        # evaluate
        k = cohen_kappa_score(_y, pred_val_class)

        # save
        try:
            _p = parse.parse(fmt, fname).named
            fs, sub, rms = _p["fs"], _p["sub"], _p["rms"]
        except:
            fs = sub = rms = None

        result.append([fname, sub, fs, rms, k])

    result_df = pd.DataFrame(result, columns=["fname", "sub", "fs", "rms", "k"])
    result_df.to_csv(os.path.join(mpath, "result_val.csv"), index=False)


if __name__ == "__main__":
    # Setup parameters and variables
    args = get_arguments()

    n_avail_cores = multiprocessing.cpu_count()
    n_jobs = args.workers if not (args.workers is None) else n_avail_cores

    db_path = os.path.join(args.root, args.dataset)
    cache_path = f"{db_path}.cache"

    # load config
    with open("config.json", "r") as f:
        config = json.load(f)
    with open("%s/db_config.json" % db_path, "r") as f:
        db_config = json.load(f)
        config["geom"] = db_config["geom"]

    if not os.path.exists(cache_path):
        # extract features and save cache
        _extract_features(args.root, args.dataset)

    ft_mask = np.in1d(ft_all, config["features"])

    # Load features and train IRF
    model, model_dir = train(cache_path, ft_mask)

    # Evaluate
    evaluate(model, cache_path, model_dir, ft_mask)
