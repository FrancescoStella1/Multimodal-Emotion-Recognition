import sys

import pickle
import numpy as np
from scipy import signal, integrate
from scipy.fft import fftshift
from scipy.signal.signaltools import lfilter

from utils import plot

PLOTS = True

BASELINE_SAMPLES = 128*3
THETA = ['Theta', 4, 8]
ALPHA = ['Alpha', 8, 12]
BETA = ['Beta', 12, 30]
GAMMA = ['Gamma', 30, 45]


def butterworth_bandpass(sig, lower, higher, sampling_freq, order=5, avoid_plot=True):
    r"""
    Filters given signal with a butterworth bandpass filter.
    
    Parameters
    ----------
    sig: array_like
        Input signal with shape (channels, time-domain samples).
    lower: float
        Lower bandpss frequency.
    higher: float
        Higher bandpass frequency.
    sampling_freq: int
        Sampling frequency of sig.
    order: int
        Order of the filter (default: 5).
    avoid_plot: bool
        Avoid plotting original and filtered signals (default: True).

    Returns
    -------
    filtered: array_like
        The output of the iir filter.

    """
    nyq = 0.5*sampling_freq            # Nyquist rate
    low = lower/nyq
    high = higher/nyq
    #sos = signal.butter(order, [low,high], btype='bandpass', output='sos')      # second-order sections output
    #filtered = signal.sosfilt(sos, sig)
    b, a = signal.butter(order, [low, high], btype='bandpass', output='ba')
    filtered = lfilter(b, a, sig, axis=-1)
    if PLOTS and not avoid_plot:
        plot.plot_signal(sig[5, 300:600], "time (ms)", "Amplitude ($\mu$V)", "Original signal from FC1 channel")
        plot.plot_signal(filtered[5, 300:600], "time (ms)",
                         "Amplitude ($\mu$V)", "Filtered signal (Theta band) from FC1 channel")
    return filtered


def get_spectrogram(sig, sampling_freq, avoid_plot=True):
    r"""
    Computes spectrogram for a given signal.
    
    Parameters
    ---------
    sig: array_like
        Signal with shape (channels, time-domain samples) for which to compute spectrogram.
    sampling_freq: int
        Sampling frequency of the signal given as input.
    avoid_plot: bool
        Avoid plotting spectrogram (default: True).

    Returns
    -------
    f: array_like
        Array of the frequencies of the spectrogram.
    log_Sxx: array_like
        Log of the Spectrogram of the signal.
    """
    f, t, Sxx = signal.spectrogram(sig, fs=sampling_freq, window='hamming', nperseg=sampling_freq,
                                   noverlap=sampling_freq//2, mode='psd')
    log_Sxx = np.log10(Sxx)
    
    if PLOTS and not avoid_plot:
        print("Shape of log Sxx: ", log_Sxx.shape)
        print("Shape of f: ", f.shape)
        print("Shape of t: ", t.shape)
        print("Shape of Sxx: ", Sxx.shape)
        plot.plot_spectrogram(t, fftshift(f), fftshift(Sxx))
    
    return f, log_Sxx


def get_stft(sig, band, sampling_freq=128, avoid_plot=True):
    r"""
    Computes the Short-Time Fourier Transform (STFT) of the given signal.

    Parameters
    ----------
    sig: array_like
        Signal with shape (channels, time-domain samples) for which to compute the STFT.
    band: str
        Band of the original signal for which the STFT is being computed.
    sampling_freq: int
        Sampling frequency of the signal (default: 128).
    avoid_plot: bool
        Avoid plotting the computed STFT.

    Returns
    -------
    freqs: array_like
        Array of the frequencies of the STFT.
    times: array_like
        Array of the time steps of the STFT.
    Zxx: array_like
        STFT of the signal.
    """
    freqs, times, Zxx = signal.stft(sig, fs=sampling_freq, window='hamming', nperseg=sampling_freq,
                                    noverlap=sampling_freq//2)
    if PLOTS and not avoid_plot:
        plot.plot_stft(freqs, times, Zxx, "STFT Magnitude - {} band".format(band))
    return freqs, times, Zxx


def get_log_psd(sig, sampling_freq, avoid_plot=True):
    r"""
    Computes the base 10 logarithm of the Power Spectral Density of the given signal.

    Parameters
    ----------
    sig: array_like
        Input signal with shape (channels, time-domain samples).
    sampling_freq: int
        Sampling frequency of the signal.
    avoid_plot: boolean
        Boolean indicating to avoid plots (default: True).

    Returns
    -------
    freqs: array_like
        Frequencies associated to the computed PSD.
    log_psd: array_like
        Base 10 logarithm of the PSD.
    """
    # Welch periodogran with 1-second window, default noverlap set to nperseg//2
    freqs, psd = signal.welch(sig, sampling_freq, nperseg=sampling_freq, noverlap=sampling_freq//2)
    log_psd = np.log10(psd)
    if PLOTS and not avoid_plot:
        plot.plot_psd(freqs, psd[0, :], "Frequency (Hz)", r"Log$_{10}$ of PSD ($\mu$V$^2$/Hz)", "Welch's Periodogram")
    return freqs, log_psd


def get_psd(sig, sampling_freq, avoid_plot=True):
    r"""Computes the logarithm of the Power Spectral Density of the given signal.

    Parameters
    ----------
    sig: array_like
        Input signal with shape (channels, time-domain samples).
    sampling_freq: int
        Sampling frequency of the signal.
    avoid_plot: boolean
        Boolean indicating to avoid plots (default: True).

    Returns
    -------
    freqs: array_like
        Frequencies associated to the computed PSD.
    psd: array_like
        Power Spectral Density of the signal with shape (channels, frequencies).
    """
    # Welch periodogran with 1-second window, default noverlap set to nperseg//2
    freqs, psd = signal.welch(sig, sampling_freq, nperseg=sampling_freq, noverlap=sampling_freq//2)
    if PLOTS and not avoid_plot:
        plot.plot_psd(freqs, psd[0, :], "Frequency (Hz)", r"Power Spectral Density ($\mu$V$^2$/Hz)",
                      "Welch's Periodogram")
    return freqs, psd


def get_rpsd(psd_bands):
    r"""
    Computes Relative Power Spectral Density (RPSD) for each channel and for each band (Theta, Alpha, Beta and Gamma).
    
    Parameters
    ----------
    psd_bands: dict
        Dict containing band names as keys and 3D arrays of shape [channels x log of PSD x trials] as values.

    Returns
    -------
    rpsd_bands: dict
        Dict containing band names as keys and 3D arrays of shape (channels, RPSD, trials) as values.
    """
    rpsd_bands = {}
    for trial in range(40):
        tmp_bands = {}
        for channel in range(32):
            psd_theta = integrate.romb(psd_bands['Theta'][channel, :, trial])
            psd_alpha = integrate.romb(psd_bands['Alpha'][channel, :, trial])
            psd_beta = integrate.romb(psd_bands['Beta'][channel, :, trial])
            psd_gamma = integrate.romb(psd_bands['Gamma'][channel, :, trial])
            psd_total = psd_theta + psd_alpha + psd_beta + psd_gamma
            if len(tmp_bands.keys()) == 0:
                tmp_bands['Theta'] = psd_theta/psd_total
                tmp_bands['Alpha'] = psd_alpha/psd_total
                tmp_bands['Beta'] = psd_beta/psd_total
                tmp_bands['Gamma'] = psd_gamma/psd_total
            else:
                tmp_bands['Theta'] = np.row_stack((tmp_bands['Theta'], psd_theta/psd_total))
                tmp_bands['Alpha'] = np.row_stack((tmp_bands['Alpha'], psd_alpha/psd_total))
                tmp_bands['Beta'] = np.row_stack((tmp_bands['Beta'], psd_beta/psd_total))
                tmp_bands['Gamma'] = np.row_stack((tmp_bands['Gamma'], psd_gamma/psd_total))
        
        if len(rpsd_bands.keys()) == 0:
            rpsd_bands = tmp_bands
        else:
            for band in ['Theta', 'Alpha', 'Beta', 'Gamma']:
                rpsd_bands[band] = np.dstack((rpsd_bands[band], tmp_bands[band]))
    return rpsd_bands


def get(filepath, avoid_plot=True, feature_type='rpsd', sampling_freq=128):
    r"""
    Performs preprocessing on the EEG signals collected from the 40 trials related to a subject.
    First, applies a butterworth bandpass filter in order to retrieve Theta, Alpha, Beta and Gamma
    sub-bands of the original EEG signals. Then, standardizes the filtered signals and extracts
    Power Spectral Densities for each sub-band and computes their logarithms.

    Parameters
    ----------
    filepath: str
        Path to the .dat file from which to retrieve EEG data.
    avoid_plot: bool
        Avoid plotting (default: True).
    feature_type: str
        Type of features to compute. It can be one between:
            |- 'log_psd': to compute the base 10 logarithm of the Power Spectral Density of each EEG band.
            |- 'psd': to compute Power Spectral Density of each EEG band.
            |- 'rpsd': to compute the Relative Power Spectral Density for each channel of each EEG band (Default value).
            |- 'spectrogram': to compute the spectrogram for each EEG band.
            |- 'stft': to compute the Short-Time Fourier Transform of each EEG band.
    sampling_freq: int
        Sampling frequency of the collected EEG signal (default: 128).

    Returns
    -------
    freqs: array_like
        Array containing the frequencies associated with the computed PSD of each sub-band.
    bands: dict
        Dict {sub-band name : 3D array}, where 3D array
        is organized as (channels, EEG feature values, trials).
    labels: array_like
        Array of shape (trials, labels) containing the annotated labels for each trial.
    """
    try:
        data = pickle.load(open(filepath, 'rb'), encoding='latin1')
        bands = {}
        for tr in range(40):
            signal = data['data'][tr][:32][:, BASELINE_SAMPLES:]                          # remove baseline samples
            # filt_bands = None
            for band in [THETA, ALPHA, BETA, GAMMA]:
                filtered = butterworth_bandpass(signal, band[1], band[2], sampling_freq, order=8, avoid_plot=avoid_plot)
                '''
                if filt_bands is None: filt_bands = filtered
                else:
                    filt_bands = np.row_stack((filt_bands, filtered))
                '''
                if feature_type == 'log_psd':
                    freqs, eeg_ft = get_log_psd(filtered, sampling_freq, avoid_plot=avoid_plot)
                if feature_type == 'psd' or feature_type == 'rpsd':
                    freqs, eeg_ft = get_psd(filtered, sampling_freq, avoid_plot=avoid_plot)
                if feature_type == 'spectrogram':
                    freqs, eeg_ft = get_spectrogram(filtered, sampling_freq, avoid_plot=avoid_plot)
                if feature_type == 'stft':
                    freqs, times, eeg_ft = get_stft(filtered, band[0], avoid_plot=avoid_plot)

                if band[0] not in bands.keys():
                    bands[band[0]] = eeg_ft

                else:
                    bands[band[0]] = np.dstack((bands[band[0]], eeg_ft))
                
                avoid_plot = True
            '''
            if not avoid_plot and PLOTS:
                plot.plot_bands(filt_bands[0], filt_bands[1], filt_bands[2], filt_bands[3], 'time (s)',
                                "Amplitude ($\mu$V)", "Bands of the signal")
            '''
        if feature_type == 'rpsd':
            bands = get_rpsd(bands)

        return freqs, bands, data['labels']
        
    except Exception as e:
        print(repr(e), file=sys.stderr)
        sys.exit(-1)
