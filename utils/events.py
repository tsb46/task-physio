import numpy as np
import pandas as pd

from nilearn.glm.first_level import glover_hrf
from patsy import dmatrix
from scipy.interpolate import interp1d


def boxcar(event_df, basis_type, tr, resample_tr, 
           n_scans, slicetime_ref, impulse_dur=0.5):
    # get time samples of functional scan based on slicetime reference
    frametimes = np.linspace(slicetime_ref, 
                             (n_scans - 1 + slicetime_ref) * tr, n_scans)
    # Create index based on resampled tr
    h_frametimes = np.arange(0, frametimes[-1]+1, resample_tr) 
    # Grab onsets from event_df
    onsets = event_df.onset.values
    # if basis is spline, create unit impulse, else, use duration values from event df
    if basis_type == 'spline':
        block_dur = impulse_dur
    else:
        block_dur = event_df.duration.values
    # initialize zero vector for event time course
    event_ts = np.zeros_like(h_frametimes).astype(np.float64)
    tmax = len(h_frametimes)
    # Get samples nearest to onsets
    t_onset = np.minimum(np.searchsorted(h_frametimes, onsets), tmax - 1)
    for t in t_onset:
        event_ts[t] = 1

    t_offset = np.minimum(np.searchsorted(h_frametimes, onsets + block_dur), tmax - 1)
    for t in zip(t_offset):
        event_ts[t] -= 1
        
    event_ts = np.cumsum(event_ts)

    return event_ts, t_onset, frametimes, h_frametimes


def convolve_regressor(event_ts, basis_type, basis, resample_tr, 
                       t_onset, event_lag, max_dur):
    # Convolve basis with event time course
    # Modified from nilearn/nilearn/glm/first_level/hemodynamic_models.py 
    if basis_type == 'spline':
        # convert to pandas series
        event_ts = pd.Series(event_ts)
        # Intialize regressor matrix
        regressor_mat = np.zeros((len(event_ts), basis.shape[1]))
        basis_dur_len = int(max_dur/resample_tr)
        # Create vector of lags
        lag_vec = np.arange(basis_dur_len).astype(int)
        # Lag event_ts by lags in lag_vec
        lag_mat = pd.concat([event_ts.shift(l, fill_value=0) for l in lag_vec], axis=1).values
        # Loop through splines bases and multiply with lagged event time course
        for l in np.arange(basis.shape[1]):
            regressor_mat[:, l] = np.dot(lag_mat, basis.iloc[:,l].values)

    elif basis_type == 'hrf':
        regressor_mat = np.array([np.convolve(event_ts, basis.iloc[:,h])[:event_ts.size]
                                 for h in range(basis.shape[1])]).T
    return regressor_mat


def create_regressor(event_df, basis, basis_type, tr, n_scans, slicetime_ref, 
                     max_dur=None, task_nknots=None, event_lag = 0, resample_tr = 0.01):
    """
    Core function for constructing fMRI regressors from task events

    This function takes a supplied BIDS event dataframe and basis, and construct a
    task fMRI regressor ready for regression

    Much of the code is modified from:
    Modified from nilearn/nilearn/glm/first_level/hemodynamic_models.py 

    Parameters:
    event_df (pd.DataFrame): BIDS compliant event dataframe -
        w/ onsets, duration and trial type
    basis (pd.DataFrame; np.array): patsy dataframe representing basis or numpy array of hrf  
    basis_type (str): type of basis (can be spline, hrf and hrf3). 'spline' is a natural
        cubic spline basis with prespecified knot locations (N=6). 'hrf' is the canonical
        hrf function. 'hrf3' is the canonical hrf function with its derivative and dispersion
    tr (float): repetition time of functional scan
    n_scans (int): number of time points of functional scan
    slicetime_ref (float): the time between consecutive samples that slices are re-referenced to
    max_dur (float): maximum duration of spline basis (ignored if not spline basis)
	task_nknots (int): degrees of freedom of spline basis for event (ignored if not spline basis)
    event_lag (float): how much time (in secs) after the end of an event to model for 
        spline basis (ignored if not spline basis)
    resample_tr (float): sampling rate of high-resolution event regressor 

    Returns:
    pd.DataFrame: dataframe containing task regressors
  
    """
    
    # Loop through trial types and create regressors
    regressor_all = []
    trial_types = event_df.trial_type.unique()
    for trial_t in trial_types:
        event_df_trial = event_df.loc[event_df.trial_type == trial_t].copy()
        # Create boxcar event time course
        event_ts, onsets, frametimes, h_frametimes = boxcar(event_df_trial, basis_type, 
                                                            tr, resample_tr, n_scans, 
                                                            slicetime_ref)

        # Convolve basis with event time course
        regressor = convolve_regressor(event_ts, basis_type, basis, resample_tr, 
                                       onsets, event_lag, max_dur)
        regressor_low = interpolate_regressor(regressor, frametimes, h_frametimes)
        regressor_df = name_regressors(regressor_low, basis_type, trial_t)
        regressor_all.append(regressor_df)

    regressor_all = pd.concat(regressor_all, axis=1)
    return regressor_all


def event_basis(basis_type, tr, max_dur=None, task_nknots=None, resample_tr=0.01):
    # define basis for creating event regressors
    # Use Natural Cubic spline basis based on Patsy package
    if basis_type == 'spline':
        # Create increasing index up to max lag
        basis_dur_len = int(max_dur/resample_tr) # maximum length of basis set by max_dur (in secs)
        lag_vec = np.arange(basis_dur_len).astype(int) 
        # Create Natural Cubic Spline basis for lagged event impulse (using patsy dmatrix)
        basis = dmatrix("cr(x, df=task_nknots) - 1",
                        {"x": lag_vec}, return_type='dataframe')
        basis.columns = [f'Knot{i}' for i in range(basis.shape[1])]
    # Canonical HRF 
    elif basis_type == 'hrf':
        basis = pd.DataFrame({'hrf': glover_hrf(tr)})

    # Add metadata to basis dataframe through attrs attribute
    basis.attrs = {
        'basis_type': basis_type,
        'tr': tr,
        'resample_tr': resample_tr,
        'max_dur': max_dur # only used for spline basis
    }

    return basis


def interpolate_regressor(regressor, frametimes, h_frametimes):
    # nilearn/nilearn/glm/first_level/hemodynamic_models/_resample_regressor.py
    f = interp1d(h_frametimes, regressor.T)
    return f(frametimes).T


def name_regressors(regressor, basis_type, trial_type):
    if basis_type == 'spline':
        cols = [f'{trial_type}_K{i}' for i in range(regressor.shape[1])]
        regressor_df = pd.DataFrame(regressor, columns=cols)
    elif basis_type == 'hrf':
        regressor_df = pd.DataFrame(regressor, columns=[f'{trial_type}'])
    return regressor_df
