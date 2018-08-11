#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Raimondas Zemblys
@email: r.zemblys@tf.su.lt
"""
#%% imports
import os, sys, glob
from distutils.dir_util import mkpath
from tqdm import tqdm

import numpy as np
import pandas as pd

###
import argparse
import multiprocessing
import json
from datetime import datetime

import parse

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import cohen_kappa_score

from util_lib.etdata import ETData
from util_lib.utils import split_path
from util_lib.irf import extractFeatures, ft_all, get_i2mc

#%%
def get_arguments():
    '''Parses command line arguments
    '''
    parser = argparse.ArgumentParser(description='Eye-movement event detection '
                                     'using Random Forest.')
    parser.add_argument('root', type=str,
                        help='The path containing eye-movement data.')
    parser.add_argument('dataset', type=str,
                        help='The directory containing experiment data.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='The directory to save output.')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of workers to use.')

    args = parser.parse_args()
    return args

#%% Setup parameters and variables
args = get_arguments()

exp_output = args.output_dir if not(args.output_dir is None)\
                             else '%s_irf' % (args.dataset)
mkpath('%s/%s'%(args.root, exp_output))
db_path = '%s/%s'%(args.root, args.dataset)

n_avail_cores = multiprocessing.cpu_count()
n_jobs = args.workers if not(args.workers is None) else n_avail_cores

#load config
with open('config.json', 'r') as f:
    config = json.load(f)
with open('%s/db_config.json'%db_path, 'r') as f:
    db_config = json.load(f)
    config['geom'] = db_config['geom']

etdata = ETData()

#%%Extract features for train and val
FILES = sorted(glob.glob('%s/%s/*[train][val]*/*.npy'%(args.root,args.dataset)))
i2mc_ok = True
for fpath in tqdm(FILES[:]):
    fdir, fname = split_path(fpath)
    odir = fdir.replace(args.dataset, exp_output)

    #create output dirs
    odir_feat = '%s/feat'%(odir)
    odir_evt = '%s/evt'%(odir)
    mkpath(odir_feat)
    mkpath(odir_evt)

    spath_feat = '%s/feat_%s.npy' % (odir_feat, fname)
    spath_evt = '%s/evt_%s.npy' % (odir_evt, fname)

    #check if feature files already exist
    if not(os.path.exists(spath_feat)) or not(os.path.exists(spath_evt)):

        etdata.load(fpath)

        #remove other events
        evt_mask = np.in1d(etdata.data['evt'], config["events"])
        etdata.data['x'][~evt_mask] = np.nan
        etdata.data['y'][~evt_mask] = np.nan
        etdata.data['status'][~evt_mask] = False

        #extract features
        if 'i2mc' in config['features']:

            fpath_i2mc = '%s/i2mc/%s_i2mc.mat'%(odir, fname)
            i2mc = get_i2mc(etdata, fpath_i2mc, config['geom'])
            if i2mc is None:
                i2mc_ok = False
                continue
            else:
                config['extr_kwargs']['i2mc'] = i2mc


        irf_features, pred_mask = extractFeatures(etdata, **config['extr_kwargs'])

        #select required features
        X = irf_features[ft_all]
        X = X.view((np.float32, len(X.dtype.names)))

        y = etdata.data['evt'][pred_mask]

        #check for other events
        assert np.in1d(np.unique(y), config["events"]).all()
        #check lengths
        assert (len(X)==len(y)) and (len(X)==pred_mask.sum())

        #save
        np.save(spath_feat, X)
        np.save(spath_evt, y)

if not i2mc_ok:
    sys.exit()
#%%Load features and train IRF
FILES = sorted(glob.glob('%s/%s/train/*.npy'%(args.root, args.dataset)))

X = []
y = []
#TODO: handle different ordering
ft_mask = np.in1d(ft_all, config['features'])
for fpath in tqdm(FILES[:]):
    fdir, fname = split_path(fpath)
    fdir = fdir.replace(args.dataset, exp_output)

    path_feat = '%s/feat/feat_%s.npy' % (fdir, fname)
    path_evt = '%s/evt/evt_%s.npy' % (fdir, fname)

    _x = np.load(path_feat)[:,ft_mask]
    _y = np.load(path_evt)
    X.append(_x)
    y.append(_y)

X = np.concatenate(X)
y = np.concatenate(y)

#train IRF
clf = RandomForestClassifier(
    n_estimators=config["n_trees"],
    max_depth=None,
    class_weight='balanced_subsample',
    max_features=3,
    n_jobs=n_jobs,
    verbose=3,
)

clf.fit(X, y)

#save
print ('Saving model...')
spath = 'models/irf_%s' % datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
mkpath(spath)
joblib.dump([config["features"], clf], '%s/model.pkl'%(spath), compress=9, protocol=2)
print ('...done')

#%%evaluate
clf.set_params(verbose=0)
FILES = sorted(glob.glob('%s/%s/val/*.npy'%(args.root, args.dataset)))
fmt = 'lookAtPoint_EL_S{sub:d}_{fs:d}_{rms:.4f}{}'
result = []
for fpath in tqdm(FILES[:]):
    fdir, fname = split_path(fpath)
    fdir = fdir.replace(args.dataset, exp_output)

    path_feat = '%s/feat/feat_%s.npy' % (fdir, fname)
    path_evt = '%s/evt/evt_%s.npy' % (fdir, fname)

    #load data
    _x = np.load(path_feat)[:,ft_mask]
    _y = np.load(path_evt)

    #predict
    pred_val = clf.predict_proba(_x)
    pred_val_class = np.argmax(pred_val, axis=1)+1

    #evaluate
    k = cohen_kappa_score(_y, pred_val_class)

    #save
    try:
        _p = parse.parse(fmt, fname).named
        fs, sub, rms = _p['fs'], _p['sub'], _p['rms']
    except:
        fs = sub = rms = None

    result.append([fname, sub, fs, rms, k])
result_df = pd.DataFrame(result, columns = ['fname', 'sub', 'fs', 'rms', 'k'])
result_df.to_csv('%s/result_val.csv'%spath, index=False)
