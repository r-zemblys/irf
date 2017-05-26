# -*- coding: utf-8 -*-
"""


@author: Raimondas Zemblys
@email: r.zemblys@tf.su.lt
"""
#import sys, os, time, glob

import os, sys
from distutils.dir_util import mkpath
import argparse
import glob
import time
import itertools
import multiprocessing
from tqdm import tqdm

import numpy as np
import scipy.interpolate as interp
import scipy.signal as sg
from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage.morphology import binary_dilation

import astropy.stats as ast
#import pandas as pd

from sklearn.externals import joblib
import sklearn.metrics as metrics


import matplotlib.pylab as plt
plt.ion()

#%%

def round_up_to_odd(f):
    w = np.int32(np.ceil(f) // 2 * 2 + 1)
    w = 3 if w < 3 else w
    return w

def rolling_window(arr, window):
    shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window)
    strides = arr.strides + (arr.strides[-1],)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

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

def aggr_events(events_raw):
    events_aggr = []
    s=0
    for bit, group in itertools.groupby(events_raw):
        event_length = len(list(group))
        e=s+event_length
        events_aggr.append([s, e, bit])
        s=e
    return events_aggr

def calc_fixPos(fix, etdata):
    data=etdata.data
    ws=round_up_to_odd(50/1000.0*etdata.fs+1) 
    fix_pos=[]
    for f in fix:
        ind_s=f[0]+ws
        ind_s = ind_s if ind_s < f[1] else f[1]
        ind_e=f[1]-ws
        ind_e = ind_e if ind_e > f[0] else f[0]
        
        posx_s = np.nanmean(data[f[0]:ind_s]['x'])
        posy_s = np.nanmean(data[f[0]:ind_s]['y'])
        posx_e = np.nanmean(data[ind_e:f[1]]['x'])
        posy_e = np.nanmean(data[ind_e:f[1]]['y'])
        fix_pos.append([posx_s, posx_e, posy_s, posy_e])
    return np.array(fix_pos)
    
class ETData():
    def __init__(self, w=100, w_vel=12, w_dir=22):
        self.etdata_dtype=np.dtype([
            ('t', np.float64),
            ('x', np.float32),
            ('y', np.float32),
            ('status', np.bool),
            ('event', np.uint8)
        ])

        self.data=np.array([], dtype=self.etdata_dtype)
        self.fs=np.nan
        self.w = w
        self.w_vel=w_vel
        self.w_dir=w_dir

        self.events=None

    def find_nearest_fs(self, a):
        fs = np.array([2000, 1250, 1000, 500, #high end
                       250, 240, 200, #middle end
                       120, 75, 60, 50, 30]) #low end
        return fs.flat[np.abs(fs - a).argmin()]

    def load(self, fpath, **kwargs):
        '''
        Loads and interpolates missing data

        source - data source. Currently only 'etdata' is supported
        interp - whether to interpolate data
        '''

        if not(kwargs.has_key('source')):
            print 'ERROR LOADING'
            return()

        if kwargs['source']=='etdata':
            self.data=np.load(fpath)

        self.maskInterp = np.zeros(len(self.data), dtype=np.bool)
        if kwargs.has_key('interp') & kwargs['interp']==True:
            r=np.arange(len(self.data))
            mask=np.isnan(self.data['x']) | np.isnan(self.data['y'])
            fx=interp.PchipInterpolator(r[~mask], self.data[~mask]['x'], 
                                        extrapolate=True)
            fy=interp.PchipInterpolator(r[~mask], self.data[~mask]['y'], 
                                        extrapolate=True)
            self.data['x'][mask]=fx(r[mask])
            self.data['y'][mask]=fy(r[mask])
            self.maskInterp=mask


    def extractFeatures(self, **kwargs):
        tic=time.time()

        #find sampling rate
        fs = self.find_nearest_fs(np.median(1/np.diff(self.data['t'])))
        self.fs=fs

        #window size for spatial measures in samples
        ws=round_up_to_odd(self.w/1000.0*fs+1)

        #window size in samples for velocity calculation
        ws_vel=round_up_to_odd(self.w_vel/1000.0*fs)

        #window size in samples for direction calculation
        ws_dir=round_up_to_odd(self.w_dir/1000.0*fs)

        ws_pad=(max((ws, ws_vel, ws_dir))-1)/2
        x_padded=np.pad(self.data['x'], (ws_pad, ws_pad), 
                        'constant', constant_values=np.nan)
        y_padded=np.pad(self.data['y'], (ws_pad, ws_pad), 
                        'constant', constant_values=np.nan)

        ws_dir_pad=(ws_dir-1)/2
        x_padded_dir=np.pad(self.data['x'], (ws_dir_pad, ws_dir_pad), 'constant', constant_values=np.nan)
        y_padded_dir=np.pad(self.data['y'], (ws_dir_pad, ws_dir_pad), 'constant', constant_values=np.nan)

        x_windowed=rolling_window(x_padded, ws)
        y_windowed=rolling_window(y_padded, ws)
        dx_windowed=rolling_window(np.diff(x_padded), ws-1)
        dy_windowed=rolling_window(np.diff(y_padded), ws-1)
        x_windowed_dir=rolling_window(np.diff(x_padded_dir), ws_dir-1)
        y_windowed_dir=rolling_window(np.diff(y_padded_dir), ws_dir-1)


        #%%Extract features
        features=dict()

        #sampling rate
        features['fs']=np.ones(len(self.data))*fs


        for d, dd in zip(['x', 'y'], [x_windowed, y_windowed]):
            #difference between positions of preceding and succeding windows,
            #aka tobii feature
            means=np.nanmean(dd, axis = 1)
            meds=np.nanmedian(dd, axis = 1)
            features['mean-diff-%s'%d] = np.roll(means, -(ws-1)/2) - \
                                         np.roll(means,  (ws-1)/2)
            features['med-diff-%s'%d] = np.roll(meds, -(ws-1)/2) - \
                                        np.roll(meds,  (ws-1)/2)

            #standard deviation
            features['std-%s'%d] = np.nanstd(dd, axis=1)
            features['std-next-%s'%d] = np.roll(features['std-%s'%d], -(ws-1)/2)
            features['std-prev-%s'%d] = np.roll(features['std-%s'%d],  (ws-1)/2)

        features['mean-diff']= np.hypot(features['mean-diff-x'],
                                        features['mean-diff-y'])
        features['med-diff']= np.hypot(features['med-diff-x'],
                                       features['med-diff-y'])

        features['std'] = np.hypot(features['std-x'], features['std-y'])
        features['std-diff'] = np.hypot(features['std-next-x'], features['std-next-y']) - \
                               np.hypot(features['std-prev-x'], features['std-prev-y'])

        #BCEA
        P = 0.68 #cumulative probability of area under the multivariate normal
        k = np.log(1/(1-P))
        #rho = [np.corrcoef(px, py)[0,1] for px, py in zip(x_windowed, y_windowed)]
        rho = vcorrcoef(x_windowed, y_windowed)
        features['bcea'] = 2 * k * np.pi * \
                           features['std-x'] * features['std-y'] * \
                           np.sqrt(1-np.power(rho,2))
        features['bcea-diff'] = np.roll(features['bcea'], -(ws-1)/2) - \
                                np.roll(features['bcea'], (ws-1)/2)

        #RMS
        features['rms'] = np.hypot(np.sqrt(np.mean(np.square(dx_windowed), axis=1)),
                                   np.sqrt(np.mean(np.square(dy_windowed), axis=1)))
        features['rms-diff'] = np.roll(features['rms'], -(ws-1)/2) - \
                               np.roll(features['rms'], (ws-1)/2)

        #disp, aka idt feature
        x_range = np.nanmax(x_windowed, axis=1) - np.nanmin(x_windowed, axis=1)
        y_range = np.nanmax(y_windowed, axis=1) - np.nanmin(y_windowed, axis=1)
        features['disp'] = x_range + y_range

        #velocity and acceleration
        features['vel']=np.hypot(sg.savgol_filter(self.data['x'], ws_vel, 2, 1),
                                 sg.savgol_filter(self.data['y'], ws_vel, 2, 1))*fs

        features['acc']=np.hypot(sg.savgol_filter(self.data['x'], ws_vel, 2, 2),
                                 sg.savgol_filter(self.data['y'], ws_vel, 2, 2))*fs**2

        #direction
        self.x_windowed_dir=x_windowed_dir
        self.y_windowed_dir=y_windowed_dir

        angl=np.arctan2(y_windowed_dir, x_windowed_dir)
        features['rayleightest'] = ast.rayleightest(angl, axis=1)

        ###i2mc
        if kwargs.has_key('i2mc_root'):
            features['i2mc']=self.feat_i2mc
        else:
            features['i2mc']=np.zeros(len(self.data))

        #remove padding and nans
        mask_nans=np.any([np.isnan(values) for key, values in features.iteritems()], axis=0)
        mask_pad=np.zeros_like(self.data['x'], dtype=np.bool)
        mask_pad[:ws_pad]=True
        mask_pad[-ws_pad:]=True
        mask=mask_nans | mask_pad
        features={key: values[~mask].astype(np.float32) for key, values in features.iteritems()}

        #return features
        self.features=np.core.records.fromarrays([features[ft_incl] for ft_incl in features_incl], np.dtype(zip(features_incl, itertools.repeat(np.float32))))
        
        self.mask = mask
#        if not(self.events is None):
#            self.events=self.events[~mask]
        self.data['status'] = ~self.mask & ~self.maskInterp

        toc=time.time()

        if kwargs.has_key('print_et') and (kwargs['print_et']==True):
            print 'Feature extraction took %.3f s.'%(toc-tic)

    def evt_postProcess(self, pred, **kwargs):
        pred_mask= ~self.mask & ~self.maskInterp
        status = pred_mask
        
        events = [1, 2, 3]
        events_p = np.ones((len(self.data), len(events)))/float(len(events))
        events_p[pred_mask,:] = pred
        
        events_f = gaussian_filter1d(events_p, 1, axis=0)
        events_pp = np.hstack((np.zeros((len(events_f), 1)), events_f))
        
        #1. mark short interpolation sequences as valid, i.e. remove short interpolation (or other "invalid data") events
        thres_id_s = round_up_to_odd(kwargs['thres_id']*self.fs/1000.+1)
        status_aggr=np.array(aggr_events(pred_mask))
        events_interp = status_aggr[status_aggr[:,-1]==False]
        md = events_interp[:,1]-events_interp[:,0]
        mask_rem_interp=md<thres_id_s
        ind_rem_interp=[i for s, e in events_interp[mask_rem_interp, :2]  for i in range(s, e)]
        status[ind_rem_interp]=True
        
        ind_leave_interp=[i for s, e in events_interp[~mask_rem_interp, :2]  for i in range(s, e)]
        events_pp[ind_leave_interp, 0]=1
        events_pp[ind_leave_interp, 1:]=0  
        
        #2. merge fixations
        thres_ifa=kwargs['thres_ifa']
        thres_ifi=kwargs['thres_ifi']
        thres_ifi_s = round_up_to_odd(thres_ifi*self.fs/1000.+1)
        
        events=np.argmax(events_pp, axis=1)
        events_aggr = np.array(aggr_events(events))
        events_fix = events_aggr[events_aggr[:,-1]==1]
        fix_pos = calc_fixPos(events_fix, self)
        
        #inter-fixation amplitudes
        ifa = np.pad(np.hypot(fix_pos[:-1, 1]-fix_pos[1:, 0], fix_pos[:-1, 3]-fix_pos[1:, 2]), (0,1), 'constant', constant_values=thres_ifa+1e-5)
        #inter-fixation intervals
        ifi = np.pad(events_fix[1:,0]-events_fix[:-1,1], (0,1), 'constant', constant_values=thres_ifi_s+1)
        mask_merge_fix=(ifa<thres_ifa) & (ifi<thres_ifi_s)
        
        ind_merge_fix=[i for s, e in zip(events_fix[mask_merge_fix,1],events_fix[np.roll(mask_merge_fix, 1),0])  for i in range(s, e)]
        events_pp[ind_merge_fix, 1]=1
        events_pp[ind_merge_fix, 2:]=0
        
        #3.1 expand saccades
        thres_sd_s=kwargs['thres_sd_s'] #make saccades to be at least 3 samples
        events=np.argmax(events_pp, axis=1)
        events_aggr = np.array(aggr_events(events))
        events_sacc = events_aggr[events_aggr[:,-1]==2]
        sd = events_sacc[:,1]-events_sacc[:,0]
        mask_expand_sacc=sd<thres_sd_s
        ind_mid_sacc=(events_sacc[mask_expand_sacc][:,1]-events_sacc[mask_expand_sacc][:,0])/2+events_sacc[mask_expand_sacc][:,0]
        ind_rem_fix=[i for s, e in zip(ind_mid_sacc-(thres_sd_s/2+thres_sd_s%2), ind_mid_sacc+(thres_sd_s/2))  for i in range(s, e)]
        events_pp[ind_rem_fix, 1]=0
        events_pp[ind_rem_fix, 3]=0
        events_pp[ind_rem_fix, 2]=1
        
        #3.2 merge nearby saccades
        thres_isi=kwargs['thres_isi']
        thres_isi_s = round_up_to_odd(thres_isi*self.fs/1000.+1)
        events=np.argmax(events_pp, axis=1)
        events_aggr = np.array(aggr_events(events))
        events_sacc = events_aggr[events_aggr[:,-1]==2]
        #inter-saccade intervals
        isi = np.pad(events_sacc[1:,0]-events_sacc[:-1,1], (0,1), 'constant', constant_values=thres_isi_s+1)
        mask_merge_sacc=isi<thres_isi_s
        ind_merge_sacc=[i for s, e in zip(events_sacc[mask_merge_sacc,1],events_sacc[np.roll(mask_merge_sacc, 1),0])  for i in range(s, e)]
        events_pp[ind_merge_sacc, 2]=1
    
        #3.3. remove too short or too long saccades. 
        thres_sd_lo=kwargs['thres_sd_lo']
        thres_sd_hi=kwargs['thres_sd_hi']
        thres_sd_lo_s = round_up_to_odd(thres_sd_lo*self.fs/1000.+1)
        thres_sd_hi_s = round_up_to_odd(thres_sd_hi*self.fs/1000.+1)
        events=np.argmax(events_pp, axis=1)
        events_aggr = np.array(aggr_events(events))
        events_sacc = events_aggr[events_aggr[:,-1]==2]
        fd = events_sacc[:,1]-events_sacc[:,0]
        mask_rem_sacc=(fd<thres_sd_lo_s) | (fd>thres_sd_hi_s)
        ind_rem_sacc=[i for s, e in events_sacc[mask_rem_sacc, :2]  for i in range(s, e)]
        events_pp[ind_rem_sacc, 2]=0
    
        #4. remove unreasonable PSOs
        events=np.argmax(events_pp, axis=1)
        events_aggr = np.array(aggr_events(events))
        pso_seq=np.pad(np.diff(events_aggr[:,-1], n=2), (1,1), 'constant') #proper sequence: 2,3,1; i.e vel(seq)=1,-2; acc(seq)=-3
        mask_inv_pso=(events_aggr[:,-1]==3) & (pso_seq!=-3)
        ind_inv_pso=[i for s, e in events_aggr[mask_inv_pso, :2] for i in range(s, e)]
        events_pp[ind_inv_pso, 2:]=0 #can't be neither pso, neither saccade
        
        #5. remove too short fixations
        thres_fd=kwargs['thres_fd']
        thres_fd_s = round_up_to_odd(thres_fd*self.fs/1000.+1)
        events=np.argmax(events_pp, axis=1)
        events_aggr = np.array(aggr_events(events))
        events_fix = events_aggr[events_aggr[:,-1]==1]
        fd = events_fix[:,1]-events_fix[:,0]
        mask_rem_fix=fd<thres_fd_s
        ind_rem_fix=[i for s, e in events_fix[mask_rem_fix, :2]  for i in range(s, e)]
        events_pp[ind_rem_fix, 0]=1
        events_pp[ind_rem_fix, 1:]=0
        
        #6.1 blink detection: remove saccade-like events between missing data
        events=np.argmax(events_pp, axis=1)
        events_aggr = np.array(aggr_events(events))
        blink_seq=np.pad(np.diff(events_aggr[:,-1], n=2), (1,1), 'constant') #blink sequence: 2,0,2; i.e vel(seq)=-2,2; acc(seq)=4
        mask_blink=(events_aggr[:,-1]==0) & (blink_seq==4)
        mask_blink=binary_dilation(mask_blink)
        ind_blink=[i for s, e in events_aggr[mask_blink, :2] for i in range(s, e)]
        events_pp[ind_blink, 0]=1
      
        #6.2 remove PSOs again, because some of saccades might been removed
        events=np.argmax(events_pp, axis=1)
        events_aggr = np.array(aggr_events(events))
        pso_seq=np.pad(np.diff(events_aggr[:,-1], n=2), (1,1), 'constant') #proper sequence: 2,3,1; i.e vel(seq)=1,-2; acc(seq)=-3
        mask_inv_pso=(events_aggr[:,-1]==3) & (pso_seq!=-3)
        ind_inv_pso=[i for s, e in events_aggr[mask_inv_pso, :2] for i in range(s, e)]
        events_pp[ind_inv_pso, 2:]=0 #can't be neither pso, neither saccade
        
        #7. Final events
        events=np.argmax(events_pp, axis=1)
        self.data['event']=events
        
    def plot(self, **kwargs):
        import matplotlib.gridspec as gridspec
        import seaborn as sns

        gs = gridspec.GridSpec(2, 2)
        ax_x = plt.subplot(gs[0, 0])
        ax_y = plt.subplot(gs[1, 0])
        ax_xy = plt.subplot(gs[:, 1])

        ax_x.plot(self.data['t'], self.data['x'])
        ax_y.plot(self.data['t'], self.data['y'])
        ax_xy.plot(self.data['x'], self.data['y'])
        ax_xy.axes.axis('equal')

        if kwargs.has_key('xlim'):
            ax_x.set_ylim(kwargs['xlim'])
            ax_xy.set_xlim(kwargs['xlim'])
        if kwargs.has_key('ylim'):
            ax_y.set_ylim(kwargs['ylim'])
            ax_xy.set_ylim(kwargs['ylim'])
        sns.despine()

#%%

DATA_DIRECTORY = './data'
OUTPUT_DIRECTORY = './events'
DATA_TYPE = 'npy'

def get_arguments():
    parser = argparse.ArgumentParser(description='Eye-movement event detection'
                                     'using Random Forest.')
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY,
                        help='The directory containing eyemovement data.')
    parser.add_argument('--data_type', type=str, default=DATA_TYPE,
                        help='Data type: npy or txt.')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIRECTORY,
                        help='The directoryto save events.')
    parser.add_argument('--clf', type=str, default=None,
                        help='Classifier.')
    parser.add_argument('--cores', type=int, default=None,
                        help='Number of cores to use for prediction.')
    return parser.parse_args()

#%% Setup parameters

args = get_arguments()
if args.clf is None:
    print 'Classifier not provided'
    sys.exit()

if not(os.path.exists(args.output_dir)):
    mkpath(args.output_dir)
n_avail_cores = multiprocessing.cpu_count()
n_cores = args.cores if not(args.cores is None) else n_avail_cores

features_incl=[
    'fs',
    'disp',
    'vel',
    'acc',
    'mean-diff',
    'med-diff',
    'rms',
    'std',
    'bcea',
    'rms-diff',
    'std-diff',
    'bcea-diff',
    'rayleightest',
    'i2mc',
]
#%%


clf, ft = joblib.load('%s'%(args.clf))
clf.set_params(n_jobs=1)

#TODO: move to external file
load_kwargs={'source':'etdata', 'interp':False}
extr_kwargs={'print_et': False}
pp_kwargs={'thres_id':75.0, #ms
           'thres_ifa':0.5, #degrees
           'thres_ifi':75., #ms
           'thres_sd_s':3, #min samples for saccades
           'thres_isi': 25., #intersaccade interval, ms
           'thres_sd_lo':6., #min saccade dur, ms
           'thres_sd_hi':150., #max saccade dur, ms
           'thres_fd':50. #min fixation duration, ms   
      }

FILES=sorted(glob.glob('%s/*.%s'%(args.data_dir, args.data_type)))
scores = []
scores_raw = []
for fpath in tqdm(FILES):
    fdir, fname = os.path.split(os.path.splitext(fpath)[0])

    etdata=ETData()
    etdata.load(fpath, **load_kwargs)
    events_org = np.copy(etdata.data['event'])
    t=time.time()
    etdata.extractFeatures(** extr_kwargs)
    #print time.time()-t
    
    #select required features, transform to matrix and predict 
    X=etdata.features[ft]
    X=X.view((np.float32, len(X.dtype.names)))
    pred=clf.predict_proba(X)
    t=time.time()
    etdata.evt_postProcess(pred, **pp_kwargs)
    #print time.time()-t
    np.save('%s/%s'%(args.output_dir, fname), etdata.data)
    
    mask = (events_org == 0) | (events_org > 3) | (etdata.data['event'] == 0)
    scores.append(metrics.cohen_kappa_score(events_org[~mask], etdata.data['event'][~mask]))
    scores_raw.append(metrics.cohen_kappa_score(events_org, etdata.data['event']))