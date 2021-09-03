#!/usr/bin/env python

import os, sys, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ndtri

def read_hdf(h5file):

    input_key = 'vortex_perturbation_parameters'
    input_df = pd.read_hdf(h5file,input_key)

    output_key = 'zeta_max'
    output_df = pd.read_hdf(h5file,output_key)

    lons = output_df['x']
    lats = output_df['y']
    lons_lats_list = list(zip(lons, lats))

    nens = len(input_df)
    ngrid = len(lons_lats_list)
    dim = len(input_df.columns)
    print('Parameter dimensionality is ', dim)
    print('Ensemble size is ', nens)
    print('Spatial grid size is ', ngrid)

    # Convert to proper numpy (there must be a cute pandas command to do this in a line or two...)
    pinput = np.empty((0, dim))
    output = np.empty((0, ngrid))
    for iens in range(nens):
        sample_key = 'vortex_4_variable_perturbation_'+str(iens+1)
        pinput = np.append(pinput, input_df.loc[sample_key+'.json'].to_numpy().reshape(1, -1), axis=0)
        output = np.append(output, output_df[sample_key].to_numpy().reshape(1, -1), axis=0)

    # Set NaNs to zero
    output = np.nan_to_num(output)
    
    print('Shape of parameter input is ', pinput.shape)
    print('Shape of model output is ', output.shape)
  
    return pinput, output    

def KL(data):
    # data is ngrid x nens
    mean = np.mean(data, axis=1)
    cov = np.cov(data)
    ngrid = data.shape[0]

    # Set trapesoidal rule weights
    weights = np.ones(ngrid)
    weights[0] = 0.5
    weights[-1] = 0.5
    weights = np.sqrt(weights)

    cov_sc = np.outer(weights, weights) * cov

    eigval, eigvec = np.linalg.eigh(cov_sc)

    kl_modes = eigvec / weights.reshape(-1, 1) # ngrid, neig
    eigval[eigval<1.e-14] = 1.e-14



    tmp = kl_modes[:, ::-1] * np.sqrt(eigval[::-1])
    rel_diag = (np.cumsum(tmp * tmp, axis=1) + 0.0) / (np.diag(cov).reshape(-1, 1) + 0.0)

    xi = np.dot(data.T - mean, eigvec * weights.reshape(-1, 1)) / np.sqrt(eigval) #nens, neig

    xi = xi[:, ::-1]
    kl_modes = kl_modes[:, ::-1]
    eigval = eigval[::-1]

    plt.figure(figsize=(12,9))
    plt.plot(range(1,ngrid+1),eigval, 'o-')
    plt.gca().set_xlabel('x')
    plt.gca().set_ylabel('Eigenvalue')
    plt.savefig('eig.png')
    plt.gca().set_yscale('log')
    plt.savefig('eig_log.png')
    plt.close()

    plt.figure(figsize=(12,9))
    plt.plot(range(ngrid),mean, label='Mean')
    for imode in range(ngrid):
        plt.plot(range(ngrid),kl_modes[:,imode], label='Mode '+str(imode+1))
    plt.gca().set_xlabel('x')
    plt.gca().set_ylabel('KL Modes')
    plt.legend()
    plt.savefig('KLmodes.png')
    plt.close()

    return mean, kl_modes, eigval, xi, rel_diag, weights

##############################################################################
# MAIN SCRIPT ################################################################
##############################################################################

h5name = 'florence_40member.h5'

# Load the input/outputs
pinput, output = read_hdf(h5name)

# Transform the uniform dimension into gaussian
pinput[:, 2] = ndtri((pinput[:, 2]+1.)/2.)
# Output this to text to be used in UQtk function
np.savetxt('xdata.dat', pinput) #because pce_eval expects xdata.dat as input

# adjusting the output to have less nodes for now (every 100 points)
ymodel = output[:, ::100].T # ymodel has a shape of ngrid x nens
ngrid, nens = ymodel.shape

## Evaluating the KL modes
# mean is the average field, size (ngrid,)
# kl_modes is the KL modes ('principal directions') of size (ngrid, ngrid)
# eigval is the eigenvalue vector, size (ngrid,)
# xi are the samples for the KL coefficients, size (nens, ngrid)
mean, kl_modes, eigval, xi, rel_diag, weights = KL(ymodel)

# pick the first neig eigenvalues, look at rel_diag array or eig.png to choose how many eigenmodes you should pick without losing much accuracy
# can go all the way to neig = ngrid, in which case one should exactly recover ypred = ymodel
neig = 25
xi = xi [:, :neig]
eigval = eigval[:neig]
kl_modes = kl_modes[:, :neig]

# Evaluate KL expansion using the same xi.
#
# WHAT NEEDS TO BE DONE: pick each column of xi (neig of them) and build PC surrogate for it like in run_pc.py (or feed the xi matrix to uqpc/uq_pc.py which I think Zach has looked at?), and then replace the xi below with its PC approximation xi_pc. Depends on your final goals, but the surrogate xi_pc and the associated ypred can be evaluated a lot more than 40 times and can be used for sensitivity analysis, moment extraction and model calibration. Essentially you will have a KL+PC spatiotemporal surrogate approximation of your model.
#
ypred = mean + np.dot(np.dot(xi, np.diag(np.sqrt(eigval))), kl_modes.T)
ypred = ypred.T
# now ypred is ngrid x nens just like ymodel

# Plot to make sure ypred and ymodel are close
plt.plot(ymodel, ypred, 'o')
plt.show()

# Pick a QoI of interest, for example, the mean of the whole region
qoi = np.mean(output, axis=1)

np.savetxt('qoi.dat', qoi)
# Builds second order PC expansion for the QoI
uqtk_cmd = 'regression -x xdata.dat -y qoi.dat -s HG -o 2 -l 0'
os.system(uqtk_cmd)

# Evaluates the constructed PC at the input for comparison
uqtk_cmd = 'pce_eval -f coeff.dat -s HG -o 2'
os.system(uqtk_cmd)
qoi_pc = np.loadtxt('ydata.dat')

# shows comparison of predicted against "real" result
plt.plot(qoi, qoi_pc, 'o')
plt.plot([0,1],[0,1], 'k--', lw=1)
plt.show()
