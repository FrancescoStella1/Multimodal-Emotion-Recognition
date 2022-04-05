import matplotlib.pyplot as plt
import numpy as np

FONTSIZE = 14
FONTSIZE2 = 12


def plot_bands(theta_band, alpha_band, beta_band, gamma_band, xlabel, ylabel, title):
    r"""
    Plots the bands related to a signal.

    Parameters
    ----------
    theta_band: array_like
        Theta band of the original signal.
    alpha_band: array_like
        Alpha band of the original signal.
    beta_band: array_like
        Beta band of the original signal.
    gamma_band: array_like
        Gamma band of the original signal.
    xlabel: str
        X-axis label.
    ylabel: str
        Y-axis label.
    title: str
        Title of the plot.
    """
    THETA = ['Theta', theta_band]
    ALPHA = ['Alpha', alpha_band]
    BETA = ['Beta', beta_band]
    GAMMA = ['Gamma', gamma_band]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    for sig, ax in zip([THETA, ALPHA, BETA, GAMMA], [ax1, ax2, ax3, ax4]):
        t = np.linspace(0, np.size(theta_band), np.size(theta_band), False)
        ax.plot(t, sig[1])
        ax.set_title(sig[0], fontsize=FONTSIZE2)
        ax.axis([0, np.size(sig[1]), np.min(sig[1])-2, np.max(sig[1])+2])
        ax.set_xlabel(xlabel, fontsize=FONTSIZE2)
        ax.set_ylabel(ylabel, fontsize=FONTSIZE2)
    plt.title(title, fontsize=FONTSIZE)
    plt.tight_layout()
    plt.show()


def plot_corr_matrix(corr_mat):
    r"""
    Plots the correlation matrix given as input.

    Parameters
    ----------
    corr_mat: array_like
        Correlation matrix to plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    m_ax = ax.matshow(corr_mat)
    fig.colorbar(m_ax)
    plt.show()


def plot_psd(freqs, power, xlabel, ylabel, title):
    r"""
    Plots the Power Spectral Density (PSD) given as input.
    
    Parameters
    ----------
    freqs: array_like
        Array of the frequencies.
    power: array_like
        Array of the power values.
    xlabel: str
        X-axis label.
    ylabel: str
        Y-axis label.
    title: str
        Title of the plot.
    """
    fig, ax = plt.subplots(1, 1)
    if len(power.shape)>1:
        for idx, p in enumerate(power):
            ax.plot(freqs, p, label='channel_{}'.format(idx+1))
        plt.legend()
    else:
        ax.plot(freqs, power)
    ax.set_title(title, fontsize=FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    plt.title(title, fontsize=FONTSIZE)
    plt.tight_layout()
    plt.show()


def plot_rfecv(rfecv, folds, min_features, scoring):
    r"""
    Plots the scores of the cross validation of the RFE algorithm.

    Parameters
    ----------
    rfecv: object
        Sklearn RFECV object that manages the feature elimination procedure.
    folds: int
        Number of folds to apply in CV.
    min_features: int
        Minimum number of features to preserve.
    scoring: str
        Evaluation criterion used in CV.
    """
    plt.figure()
    plt.xlabel("Number of features selected", fontsize=FONTSIZE)
    plt.ylabel(f"Cross validation score {scoring}", fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE2)
    plt.yticks(fontsize=FONTSIZE2)
    plt.title(f"RFECV Scores ({folds}-Folds)", fontsize=FONTSIZE)
    plt.ylim(0, 1.5)
    plt.plot(range(min_features, len(rfecv.grid_scores_) + min_features), rfecv.grid_scores_)
    plt.tight_layout()
    plt.show()


def plot_signal(sig, xlabel, ylabel, title):
    r"""Plots the signal given as input.
    
    Parameters
    ----------
    sig: array_like
        Signal to plot.
    xlabel: str
        X-axis label.
    ylabel: str
        Y-axis label.
    title: str
        Title of the plot.    
    """
    fig, ax = plt.subplots(1, 1)
    t = np.linspace(0, np.size(sig), np.size(sig))
    ax.plot(t, sig)
    ax.set_title(title, fontsize=FONTSIZE)
    ax.axis([0, np.size(sig), np.min(sig)-2, np.max(sig)+2])
    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE2)
    plt.yticks(fontsize=FONTSIZE2)
    plt.tight_layout()
    plt.show()


def plot_spectrogram(time, freqs, psd):
    r"""Plots the spectrogram given as input.
    
    Parameters
    ----------
    time: array_like
        Array of the time steps.
    freqs: array_like
        Array of the frequencies.
    psd: array_like
        Array of the PSD.
    """
    fig, ax = plt.subplots(1, 1)
    ax.pcolormesh(time, freqs, psd[0, :, :], shading='gouraud')
    plt.xlabel("Time (s)", fontsize=FONTSIZE)
    plt.ylabel("Frequency (Hz)", fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE2)
    plt.yticks(fontsize=FONTSIZE2)
    plt.title("Spectrogram", fontsize=FONTSIZE)
    plt.tight_layout()
    plt.show()


def plot_stft(freqs, times, Zxx, title, fs=128):
    r"""Plots the Short-Time Fourier Transform (STFT) of a given signal.
    
    Parameters
    ----------
    freqs: array_like
        Array of the frequencies.
    times: array_like
        Array of the time steps.
    Zxx: array_like
        STFT of the signal.
    title: str
        Title of the plot.
    fs: int
        Sampling frequency of the original signal (default: 128).
    """
    plt.pcolormesh(times, freqs, np.abs(Zxx[0, :, :]), vmin=0, shading='gouraud')
    plt.title(title, fontsize=FONTSIZE)
    plt.xlabel('Time (s)', fontsize=FONTSIZE)
    plt.ylabel('Frequency (Hz)', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE2)
    plt.yticks(fontsize=FONTSIZE2)
    plt.show()


def plot_pca_comp_stats(x, y, xlabel, ylabel1, ylabel2, title):
    r"""Plots Mean Squared Error (MSE) and Explained Variance (EV) in function of PCA components.
    
    Parameters
    ----------
    x: array_like
        Array of the number of components used to computed the PCA.
    y: array_like
        Array containing tuples (MSE, EV) for each element of x.
    xlabel: str
        X-axis label.
    ylabel1: str
        First y-axis label, related to the MSE.
    ylabel2: str
        Second y-axis label, related to EV.
    title: str
        Title of the plot.
    """
    fig, ax1 = plt.subplots(1, 1)
    ax1.set_xticks(x)
    ax2 = ax1.twinx()
    mse = list(map(lambda y: y[0], y))
    var = list(map(lambda y: y[1], y))
    ax1.plot(x, mse, color='orange', label='mean squared error')
    ax1.set_xlabel(xlabel, fontsize=FONTSIZE2)
    ax1.set_ylabel(ylabel1, fontsize=FONTSIZE2)
    
    ax2.plot(x, var, color='cyan', label='explained variance')
    ax2.set_xlabel(xlabel, fontsize=FONTSIZE2)
    ax2.set_ylabel(ylabel2, fontsize=FONTSIZE2)
    ax1.set_title(title, fontsize=FONTSIZE)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.xticks(fontsize=FONTSIZE2)
    plt.yticks(fontsize=FONTSIZE2)
    plt.tight_layout()
    plt.show()
