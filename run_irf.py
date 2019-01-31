#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Raimondas Zemblys
@email: r.zemblys@tf.su.lt
"""
#%% imports
import os, sys, copy
from distutils.dir_util import mkpath
from tqdm import tqdm

import numpy as np
import pandas as pd

#import seaborn as sns
#sns.set_style("ticks")

###
import argparse, multiprocessing, json, fnmatch
from datetime import datetime

from sklearn.externals import joblib

from util_lib.etdata import ETData
from util_lib.utils import split_path
from util_lib.irf import extractFeatures, get_i2mc
from util_lib.irf import postProcess
from util_lib.irf import hpp

#%%
def get_arguments():
    '''Parses command line arguments
    '''
    parser = argparse.ArgumentParser(description='Eye-movement event detection '
                                     'using Random Forest.')
    parser.add_argument('clf', type=str,
                        default='irf_2018-03-26_20-46-41',
                        help='Classifier')
    parser.add_argument('root', type=str,
                        help='The path containing eye-movement data')
    parser.add_argument('dataset', type=str,
                        help='The directory containing experiment data')
    parser.add_argument('--ext', type=str, default='npy',
                        help='File type')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='The directory to save output')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of workers to use')
    parser.add_argument('--save_csv', action='store_true',
                        help='Save output as csv file')

    args = parser.parse_args()
    return args

#%% Setup parameters and variables
args = get_arguments()

ROOT_OUTPUT = args.output_dir if not(args.output_dir is None)\
                              else '%s/%s_irf' % (args.root, args.dataset)
fpath_log = '%s/irf_%s.log'%(ROOT_OUTPUT, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
db_path = '%s/%s'%(args.root, args.dataset)

n_avail_cores = multiprocessing.cpu_count()
n_jobs = args.workers if not(args.workers is None) else n_avail_cores

etdata = ETData()
pp = hpp()

#%% Main
#load classifier
if args.clf is None:
    print 'Classifier not provided'
    sys.exit()
else:
    print ('Loading model...')
    ft, clf = joblib.load('models/%s/model.pkl'%(args.clf))
    clf.set_params(n_jobs=n_jobs, verbose=0)
    print ('...done')

#load config
with open('config.json', 'r') as f:
    config = json.load(f)
with open('%s/db_config.json'%db_path, 'r') as f:
    db_config = json.load(f)
    config['geom'] = db_config['geom']

#get file list and process data
FILES = []
for _root, _dir, _files in os.walk(db_path):
    FILES.extend(['%s/%s' % (_root, _file)
                  for _file in fnmatch.filter(_files, '*.%s'%args.ext)])

for fpath in tqdm(FILES):
    fdir, fname = split_path(fpath)
    odir = fdir.replace(db_path, ROOT_OUTPUT)
    mkpath(odir)

    etdata.load(fpath)
#    evt_gt = copy.deepcopy(etdata.data['evt']) #gound truth events

    #extract features
    if 'i2mc' in ft:
        fdir_i2mc = odir.replace(ROOT_OUTPUT, '%s/i2mc'%ROOT_OUTPUT)
        fpath_i2mc = '%s/%s_i2mc.mat'%(fdir_i2mc, fname)
        i2mc = get_i2mc(etdata, fpath_i2mc, config['geom'])
        if i2mc is None:
            continue
        else:
            config['extr_kwargs']['i2mc'] = i2mc

    irf_features, pred_mask = extractFeatures(etdata, **config['extr_kwargs'])
    if not(len(irf_features)):
        with open(fpath_log, 'a') as f:
            f.write('EMPTY:\t%s\n'%fpath.replace(ROOT_OUTPUT, ''))
        continue

    #select required features, transform to matrix and predict
    X = irf_features[ft]
    X = X.view((np.float32, len(X.dtype.names)))
    pred = clf.predict_proba(X)

    #probabilistic post-processing
    etdata.data['evt'], etdata.data['status'], pred_ = \
    postProcess(etdata, pred, pred_mask, **config['pp_kwargs'])

    #hard post-processing
    etdata.data['evt'], etdata.data['status'], pp_rez, pp_inds = \
    pp.run_pp(etdata, **config['pp_kwargs'])

#    pp_check.run_pp(etdata, **config['pp_kwargs'])

    #save
    spath = '%s/%s'%(odir, fname)
    etdata.save(spath)

    #save csv
    if args.save_csv:
        data_df = pd.DataFrame(etdata.data)
        data_df.to_csv('%s.csv'%spath)
