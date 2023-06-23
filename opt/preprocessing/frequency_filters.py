import pandas as pd
import numpy as np

from scipy import fftpack
from scipy import signal


def high_pass(sig, fs=1000, threshold_factor=0.25):
    """
    Applies a high-pass filter to the input signal.
    
    Parameters:
    -----------
    sig : pandas.Series or numpy.ndarray
        The input signal to be filtered.
    fs : int, optional
        The sampling frequency of the input signal. Default is 1000 Hz.
    threshold_factor : float, optional
        The fraction of the maximum signal amplitude to be used as the filter threshold.
        Default is 0.25.
        
    Returns:
    --------
    numpy.ndarray
        The filtered signal.
    """
    sig = sig.values if isinstance(sig, pd.Series) else sig
    sig[np.isnan(sig) | np.isinf(sig)] = 0  # Replace NaN and infinity values with zero
    sig = np.abs(sig.real)  # Ensure the signal is real and non-negative
    try:
        fourier = fftpack.fft(sig)
        frequencies = fftpack.fftfreq(sig.size)
        max_amp = np.abs(fourier).max()
        threshold = max_amp * threshold_factor + 1e-10  # Add a small constant value to avoid division by zero

        sos = signal.butter(1, threshold, 'hp', fs=fs, output='sos')
        filtered = signal.sosfilt(sos, sig, axis=-1, zi=None)
        return filtered
    except (ValueError, RuntimeError) as e:
        if np.isfinite(sig).all() and not np.all(sig == 0):
            raise e
        else:
            return np.zeros_like(sig)
