"""
@author: Raimondas Zemblys
@email: raimondas.zemblys@gmail.com
"""
import os
import sys
import copy
import argparse
import multiprocessing
import json
import fnmatch
import logging
from datetime import datetime

from distutils.dir_util import mkpath


from tqdm import tqdm
import joblib

import numpy as np
import pandas as pd

from util_lib.etdata import ETData
from util_lib.utils import split_path
from util_lib.irf import extract_features, get_i2mc
from util_lib.irf import post_process
from util_lib.irf import hpp


def get_arguments():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(
        description="Eye-movement event detection " "using Random Forest."
    )
    parser.add_argument("clf", type=str, default="irf_2018-03-26_20-46-41", help="Classifier")
    parser.add_argument("root", type=str, help="The path containing eye-movement data")
    parser.add_argument("dataset", type=str, help="The directory containing experiment data")
    parser.add_argument("--ext", type=str, default="npy", help="File type")
    parser.add_argument("--output-dir", type=str, default=None, help="The directory to save output")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers to use")
    parser.add_argument("--save-csv", action="store_true", help="Save output as csv file")

    return parser.parse_args()


if __name__ == "__main__":
    # Setup parameters and variables
    args = get_arguments()

    db_path = os.path.join(args.root, args.dataset)
    root_output = args.output_dir if args.output_dir is not None else f"{db_path}_irf"
    mkpath(root_output)

    _tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fpath_log = os.path.join(root_output, f"irf_{_tag}.log")

    logging.basicConfig(
        format="%(levelname)s. %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(fpath_log, mode="w"),
            logging.StreamHandler(),
        ],
    )

    n_avail_cores = multiprocessing.cpu_count()
    n_jobs = args.workers if args.workers is not None else n_avail_cores

    etdata = ETData()
    pp = hpp()

    # load classifier
    if args.clf is None:
        print("Classifier not provided")
        sys.exit()
    else:
        print("Loading model...")
        ft, clf = joblib.load(f"models/{args.clf}/model.pkl")
        clf.set_params(n_jobs=n_jobs, verbose=0)
        print("...done")

    # load config
    with open("config.json", "r") as f:
        config = json.load(f)
    with open(os.path.join(db_path, "db_config.json"), "r") as f:
        db_config = json.load(f)
        config["geom"] = db_config["geom"]

    # get file list and process data
    files = []
    for _root, _dir, _files in os.walk(db_path):
        files.extend(
            [os.path.join(_root, _file) for _file in fnmatch.filter(_files, f"*.{args.ext}")]
        )

    for fpath in tqdm(files):
        fdir, fname = split_path(fpath)
        odir = fdir.replace(db_path, root_output)
        mkpath(odir)

        etdata.load(fpath)
        evt_gt = copy.deepcopy(etdata.data["evt"])  # ground truth events

        # extract features
        if "i2mc" in ft:
            fdir_i2mc = odir.replace(root_output, "%s/i2mc" % root_output)
            fpath_i2mc = "%s/%s_i2mc.mat" % (fdir_i2mc, fname)
            i2mc = get_i2mc(etdata, fpath_i2mc, config["geom"])
            if i2mc is None:
                continue
            else:
                config["extr_kwargs"]["i2mc"] = i2mc

        irf_features, pred_mask = extract_features(etdata, **config["extr_kwargs"])
        if not len(irf_features):
            logging.error(f"File emty: {fpath}")
            continue

        # select required features, transform to matrix and predict
        x = irf_features[ft]
        x = np.array(x.tolist())
        pred = clf.predict_proba(x)

        # probabilistic post-processing
        etdata.data["evt"], etdata.data["status"], pred_ = post_process(
            etdata, pred, pred_mask, events=config.get("events"), **config["pp_kwargs"]
        )

        # hard post-processing
        etdata.data["evt"], etdata.data["status"], pp_rez, pp_inds = pp.run_pp(
            etdata, **config["pp_kwargs"]
        )

        # pp_check.run_pp(etdata, **config['pp_kwargs'])

        # save
        spath = os.path.join(odir, fname)
        etdata.save(spath)

        # save csv
        if args.save_csv:
            data_df = pd.DataFrame(etdata.data)
            data_df["gt"] = evt_gt
            data_df.to_csv(f"{spath}.csv")
