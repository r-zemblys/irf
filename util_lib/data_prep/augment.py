#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: rz
@email: r.zemblys@tf.su.lt
"""
#%% imports
import os, sys, glob, copy
from distutils.dir_util import mkpath

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
plt.rc("axes.spines", top=False, right=False)
plt.ioff()

#import seaborn as sns
#sns.set_style("ticks")

###
sys.path.append('..')

import random
import parse

import matplotlib.mlab as mlab
from scipy.interpolate import interp1d
import scipy.signal as sg

from etdata import ETData

#%%functions
def BoxMuller_gaussian(u1,u2):
  z1 = np.sqrt(-2*np.log(u1))*np.cos(2*np.pi*u2)
  z2 = np.sqrt(-2*np.log(u1))*np.sin(2*np.pi*u2)
  return z1,z2

#%% setup parameters
ROOT = '../../etdata'
EXP = 'lookAtPoint_EL'
ROOT_OUTPUT = '%s/%s/augment' % (ROOT, EXP)
mkpath(ROOT_OUTPUT)
ROOT_TRAIN = '%s/%s/training' % (ROOT, EXP)
mkpath(ROOT_TRAIN)
for _dir in ['train', 'val']:
    mkpath('%s/%s'%(ROOT_TRAIN, _dir))
ROOT_TEST = '%s/%s/testing' % (ROOT, EXP)
mkpath(ROOT_TEST)

#select subjects
subjects = [1, 2, 4, 5, 6]
random.seed(062217)
random.shuffle(subjects)
subjects_test = subjects[:1]
subjects_train = subjects[1:]

sampling_rates = [1250, 1000, 500, 300, 250, 200, 120, 60, 30]
lowpass_size = 20.0 #ms

#setup noise mapping
FWHM = 20 #approximate extent of data
xrms = 3 #rms multiplier
delta = 0.1 #resolution

#TODO: this can be optimized
N = 10 #noise levels
rms_s = 0.005
rms_levels=[0]
for i in range(N):
    rms_levels.append(round(rms_s,3))
    rms_e=rms_s*xrms
    rms_s=rms_e-rms_s

#calculates sigma for full width at \part\ maximum
w = np.hypot(FWHM, FWHM)
sigma = w*2/(2.0*np.sqrt(2*np.log(xrms)))

x = np.arange(-FWHM, FWHM, delta)
y = np.arange(-FWHM, FWHM, delta)
X, Y = np.meshgrid(x, y)
Z1 = mlab.bivariate_normal(X, Y, sigma, sigma, 0.0, 0.0)

s = np.ptp(Z1)
m = Z1.min()

##draw noise function
#plt.ion()
#rms=rms_levels[1]
#Z1*=-1
#Z1+=s+m
#Z1*=(rms*(xrms-1)/s)
#Z1+=rms
#
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(X, Y, Z1)

#%% augment data
FILES = glob.glob('%s/%s/*.npy'%(ROOT, EXP))
etdata = ETData()
np.random.seed(062217)
for fpath in FILES:
    fdir, fname = os.path.split(os.path.splitext(fpath)[0])
    sub = parse.parse('lookAtPoint_EL_S{sub:d}', fname).named['sub']

    #skip if subject is not included
    if not(sub in subjects):
        continue

    data = etdata.load(fpath)
    #interpolate to avoid artifacts when resampling
    mask_interp = np.isnan(data['x']) | np.isnan(data['y']) | (~data['status'])
    assert (mask_interp == ~data['status']).all()
    for _d in ['x', 'y']:
        _f = interp1d(data['t'][~mask_interp], data[_d][~mask_interp])
        data[_d][mask_interp] = _f(data['t'][mask_interp])

    #resample
    for fs in sampling_rates:
        print sub, fs

        if fs == 1000:
            #do not filter original data
            _data_ds = copy.deepcopy(data)
        else:
            #low-pass filter
            _data = copy.deepcopy(data)
            nyq_rate = fs / 2.
            cutoff_hz = 0.8*nyq_rate
            numtaps = np.int16(fs*lowpass_size/1000.0)
            if numtaps < 2:
                numtaps = 2

            b, a = sg.butter(numtaps, cutoff_hz/nyq_rate) #left in case of different cutoff
            _data['x'] = sg.filtfilt(b, a, _data['x'])
            _data['y'] = sg.filtfilt(b, a, _data['y'])

            #resample
            t_interp = np.arange(_data['t'][0], _data['t'][-1], 1.0/fs)
            #setup containers for resampled data
            if fs > 1000:
                _data_ds = np.zeros_like(np.hstack((_data, _data)))[:len(t_interp)]
            else:
                _data_ds = np.zeros_like(_data)[:len(t_interp)]
            _data_ds['t'] = t_interp

            for var, interp_type in zip(['x', 'y', 'status', 'evt'],
                                        ['slinear', 'slinear', 'nearest', 'nearest']):
                _f = interp1d(_data['t'], _data[var], kind=interp_type)
                _data_ds[var] = _f(_data_ds['t'])

        #add noise
        for rms in rms_levels[:]:
            _data_noise = copy.deepcopy(_data_ds)

            #iterate through samples and get noise level for each
            n = []
            for _sample in _data_noise:
                X, Y = np.meshgrid(_sample['x'], _sample['y'])
                v = mlab.bivariate_normal(X, Y, sigma, sigma, 0.0, 0.0)[0][0]
                v*=-1
                v+=s+m
                v*=(rms*(xrms-1)/s)
                v+=rms
                n.append(v)

            u1 = np.random.uniform(0,1,len(_data_noise))
            u2 = np.random.uniform(0,1,len(_data_noise))

            noise_x, noise_y = BoxMuller_gaussian(u1,u2)
            n = np.array(n)/2.0

            noise_x*=n
            noise_y*=n
            _data_noise['x']+=noise_x
            _data_noise['y']+=noise_y

            #remove interpolated samples
            _data_noise['x'][~_data_noise['status']] = np.nan
            _data_noise['y'][~_data_noise['status']] = np.nan

            #save data
            etdata.load(_data_noise, **{'source': 'array'})
            etdata.save('%s/%s_%d_%.3f'%(ROOT_OUTPUT, fname, fs, rms))

            #train/val split
            l = len(_data_noise)
            val_ind = np.random.randint(0.25*l, 0.5*l)

            #save split data
            if sub in subjects_train:
                np.save('%s/train/%s_%d_%.3f_train1'%(ROOT_TRAIN, fname, fs, rms),
                        _data_noise[:val_ind])
                np.save('%s/train/%s_%d_%.3f_train2'%(ROOT_TRAIN, fname, fs, rms),
                        _data_noise[val_ind+0.25*l:])

                np.save('%s/val/%s_%d_%.3f_val'%(ROOT_TRAIN, fname, fs, rms),
                        _data_noise[val_ind:val_ind+0.25*l])
            else:
                np.save('%s/%s_%d_%.3f_test'%(ROOT_TEST, fname, fs, rms),
                        _data_noise)
