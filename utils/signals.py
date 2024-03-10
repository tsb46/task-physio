import neurokit2 as nk
import numpy as np

from nilearn.glm.first_level.hemodynamic_models import glover_hrf
from patsy import dmatrix
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfiltfilt, sosfreqz


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butterworth_filter(signals, lowcut, highcut, fs, filter_type, npad=1000):
    # Set filter params
    sos = butter_bandpass(lowcut, highcut, fs)
    # Median padding to reduce edge effects
    signals = np.pad(signals,[(npad, npad), (0, 0)], 'median')
    # backward and forward filtering
    signals = sosfiltfilt(sos, signals, axis=0)
    # Cut padding to original signal
    signals = signals[npad:-npad, :]
    return signals


def clip_spikes(ts, spike_thres=5):
    # Clip time series within bounds
    # Spike thres is set in z-score units, must convert to original units
    ts_mean = ts.mean()
    ts_std = ts.std()
    ts_spike_thres = (spike_thres*ts_std) + ts_mean
    ts_clip = ts.clip(-ts_spike_thres, ts_spike_thres)
    return ts_clip


def physio_basis(lags, lag_df):
    # Create vector of lags
    lag_vec = np.arange(lags).astype(int)
    # create cubic spline basis
    basis = dmatrix("cr(x, df=lag_df) - 1", {"x": lag_vec}, 
                    return_type='dataframe')
    return basis


def resample_signal(signal, n_scan):
    signal_resamp = nk.signal_resample(signal, desired_length=n_scan)
    return signal_resamp



