import os, sys
import numpy as np
import pickle

from skopt import dump, load
from xgboost import XGBClassifier, XGBRFClassifier


def load_eeg_data(load_path, filename):
    r"""
    Reads EEG-related data from a .npz file.

    Parameters
    ----------
    load_path: str
        Path containing the .npz file to load.
    filename: str
        Name of the file to load.

    Returns
    -------
    freqs: array_like
        Array containing the frequencies associated with the PSD of the signal.
    psd_data: dict
        Dict containing the PSD data of each extracted band (Theta, Alpha, Beta and Gamma) from the EEG signal.
    labels: array_like
        Array containing the annotated labels for each trial.
    """
    try:
        psd_data = {}
        data = np.load(os.path.join(load_path, filename), allow_pickle=True)
        freqs = data['freqs']
        psd_data['Theta'] = data['psd_theta']
        psd_data['Alpha'] = data['psd_alpha']
        psd_data['Beta'] = data['psd_beta']
        psd_data['Gamma'] = data['psd_gamma']
        labels = data['labels']                         # Valence, Arousal, Dominance, Liking
        return freqs, psd_data, labels
    
    except Exception as e:
        print("[ Error while loading EEG data ] - {}".format(repr(e)), file=sys.stderr)
        sys.exit(1)


def load_all_eeg_data(path):
    r"""
    Loads EEG data from .npz files and returns two arrays for EEG and labels data.

    Parameters
    ---------
    path: str
        Path to the preprocessed .npz files.

    Returns
    -------
    eeg_data: array_like
        Array of the EEG data with shape (subjects x trials x EEG features, channels x bands).
    labels: array_like
        Array of the labels with shape (subject x trials x EEG features, V/A/D/L labels).
    """
    entries = os.listdir(path)
    eeg_data = None
    label_data = None
    print("Reading eeg data..")
    for entry in entries:
        if not entry.startswith('s'):
            continue
        _, psd_bands, labels = load_eeg_data(path, entry)          # (freqs, [channels, freqs, trials], labels)
        eeg = psd_bands['Theta']
        eeg = np.row_stack((eeg, psd_bands['Alpha']))
        eeg = np.row_stack((eeg, psd_bands['Beta']))
        eeg = np.row_stack((eeg, psd_bands['Gamma']))

        u_eeg = unstack_eeg(eeg)
        if eeg_data is None:
            eeg_data = u_eeg
        else:
            eeg_data = np.column_stack((eeg_data, u_eeg))
        if label_data is None:
            label_data = labels
        else:
            label_data = np.row_stack((label_data, labels))
    label_data = np.repeat(label_data, eeg_data.shape[1] / 1280, axis=0)     # align labels with EEG data
    return eeg_data.T, label_data


def load_eeg_model(load_path, filename):
    r"""
    Loads and returns EEG model.

    Parameters
    ----------
    load_path: str
        Path to the model.
    filename: str
        Name of the model without extension.

    Returns
    -------
    model: object (or None)
        Requested model (or None).
    """
    filepath = os.path.join(load_path, f"{filename}.json")
    model = None
    if 'svm' in filename and os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
    elif 'xgb_rf_clf' in filename and os.path.exists(filepath):
        model = XGBRFClassifier()
        model.load_model(filepath)
    return model


def load_ensemble_data(load_path, filename):
    r"""
    Loads EEG and Facial Features models predictions from file.

    Parameters
    ----------
    load_path: str
        Path to the .npz file.
    filename: str
        Name of the file.

    Returns
    -------
    eeg_pred: array_like
        Predicted labels by the EEG model.
    ff_pred: array_like
        Predicted labels by the Facial Features model.
    labels: array_like
        True labels.
    """
    try:
        data = np.load(os.path.join(load_path, filename), allow_pickle=True)
        eeg_pred = data['eeg_pred']
        ff_pred = data['ff_pred']
        labels = data['labels']
        return eeg_pred, ff_pred, labels

    except Exception as e:
        print("[ Error while loading ensemble data ] - {}".format(repr(e)), file=sys.stderr)


def load_ensemble_model(load_path, filename):
    r"""
    Loads and returns ensemble model.

    Parameters
    ----------
    load_path: str
        Path to the model.
    filename: str
        Name of the model.

    Returns
    -------
    model: object
        Requested model (or None).
    """
    filepath = os.path.join(load_path, filename)
    model = None
    if 'svm' in filepath:
        filepath = os.path.join(load_path, f"{filename}.pkl")
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
    elif 'xgb_rf_clf' in filepath:
        filepath = os.path.join(load_path, f"{filename}.json")
        if os.path.exists(filepath):
            model = XGBRFClassifier()
            model.load_model(f"{filepath}")
    elif 'xgb_dt_clf' in filepath:
        filepath = os.path.join(load_path, f"{filename}.json")
        if os.path.exists(filepath):
            model = XGBClassifier()
            model.load_model(f"{filepath}")
    return model


def load_fe_eeg(load_path, filename):
    r"""
    Loads EEG data on which feature elimination has been applied.

    Parameters
    ----------
    load_path: str
        Path containing the .npz file.
    filename: str
        Name of the .npz file to load.

    Returns
    -------
    eeg: array_like
        Array of the EEG data.
    labels: array_like
        Array of the labels.

    """
    try:
        data = np.load(os.path.join(load_path, filename), allow_pickle=True)
        return data['eeg'], data['labels']

    except Exception as e:
        print("[ Error while loading feature elimination EEG files ] - {}".format(repr(e)), file=sys.stderr)
        sys.exit(1)


def load_ff_data(load_path, filename, return_labels=False):
    r"""
    Loads the facial features from a .npz file.

    Parameters
    ----------
    load_path: str
        Path containing the .npz file.
    filename: str
        Name of the .npz file.
    return_labels: bool
        Tries to obtain labels from .npz file.

    Returns
    -------
    features: array_like
        Array containing the facial features.
    labels: array_like
        (Optional) Labels array.
    """
    try:
        data = np.load(os.path.join(load_path, filename), allow_pickle=True)
        features = data['features']
        if not return_labels:
            return features
        labels = data['labels']
        return features, labels

    except Exception as e:
        print("[ Error while loading facial features ] - {}".format(repr(e)), file=sys.stderr)


def load_ff_model(load_path, filename):
    r"""
    Loads and returns facial features model.

    Parameters
    ----------
    load_path: str
        Path to the model.
    filename: str
        Name of the model with extension.

    Returns
    -------
    model: object (or None)
        Requested model (or None).

    """
    filepath = os.path.join(load_path, f"{filename}")
    model = None
    if os.path.exists(filepath):
        if 'svm' in filename and os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
        elif 'xgb_rf_clf' in filename and os.path.exists(filepath):
            model = XGBRFClassifier()
            model.load_model(filepath)
    return model


def load_landmark_data(load_path, filename):
    r"""
    Loads the landmarks from a .npz file.

    Parameters
    ----------
    load_path: str
        Path containing the file to load.
    filename: str
        Name of the file containing the landmarks.

    Returns
    -------
    landmarks: array_like
        Array containing the landmarks for each extracted frame from the video clip.
    """
    try:
        landmarks = np.load(os.path.join(load_path, filename), allow_pickle=True)
        return landmarks['landmarks']

    except Exception as e:
        print("[ Error while loading landmarks ] - {}".format(repr(e)), file=sys.stderr)
        sys.exit(1)


def load_optimizer(load_path, filename):
    r"""
    Loads scikit-optimize optimizer.

    Parameters
    ----------
        load_path: str
            Path containing the file to load.
        filename: str
            Name of the .pkl file to load (included extension).

    Returns
    -------
        opt: Optimizer
            Optimizer object
    """
    try:
        filepath = os.path.join(load_path, filename)
        opt = load(filepath)
        return opt

    except Exception as e:
        print("[ Error while loading optimizer ] - {}".format(repr(e)), file=sys.stderr)
        sys.exit(1)


def save_eeg_data(save_path, filename, freqs, psd_data, labels):
    r"""
    Saves frequencies, PSDs and labels related to EEG into a .npz file.

    Parameters
    ----------
    save_path: str
        Path in which to save the file.
    filename: str
        Name of the .npz file to save.
    freqs: array_like
        Array containing the frequencies associated to the PSD of each sub-band.
    psd_data: dict
        Dict containing the data related to the PSD of each extracted band of the EEG signal.
    labels: array_like
        Annotated labels from dataset.
    """
    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.savez_compressed(os.path.join(save_path, filename), freqs=freqs, psd_theta=psd_data['Theta'],
                            psd_alpha=psd_data['Alpha'], psd_beta=psd_data['Beta'], psd_gamma=psd_data['Gamma'],
                            labels=labels)
    
    except Exception as e:
        print("[ Error while saving EEG files ] - {}".format(repr(e)), file=sys.stderr)
        sys.exit(1)


def save_ensemble_data(save_path, filename, eeg_pred, ff_pred, labels):
    r"""
    Saves EEG and Facial Features models outputs into a .npz file.

    Parameters
    ----------
    save_path: str
        Path in which to save the file.
    filename: str
        Name of the .npz file.
    eeg_pred: array_like
        Predictions of the EEG model for the training and testing sets.
    ff_pred: array_like
        Predictions of the Facial Features model for the training and testing sets.
    labels: array_like
        True labels associated to the predictions of both models.
    """
    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.savez_compressed(os.path.join(save_path, filename), eeg_pred=eeg_pred, ff_pred=ff_pred, labels=labels)

    except Exception as e:
        print("[ Error while saving ensemble files ] - {}".format(repr(e)), file=sys.stderr)


def save_fe_eeg(save_path, filename, eeg, labels):
    r"""
    Saves the EEG data on which feature elimination has been applied.

    Parameters
    ----------
    save_path: str
        Path in which to save the .npz file.
    filename: str
        Name of the .npz file.
    eeg: array_like
        Array of the EEG data with shape (subjects x EEG features x trials, remained features).
    labels: array_like
        Array of the labels with shape (subjects x EEG features x trials, single label).
    """
    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.savez_compressed(os.path.join(save_path, filename), eeg=eeg, labels=labels)

    except Exception as e:
        print("[ Error while saving feature elimination EEG files ] - {}".format(repr(e)), file=sys.stderr)
        sys.exit(1)


def save_ff_data(save_path, filename, features, labels=None):
    r"""
    Saves facial features into a .npz file.

    Parameters
    ----------
    save_path: str
        Path in which to save the facial features.
    filename: str
        Name of the .npz file.
    features: array_like
        Array containing the facial features to save.
    labels: array_like
        (Optional) Labels related to the features (default: None).
    """
    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if labels is not None:
            np.savez_compressed(os.path.join(save_path, filename), features=features, labels=labels)
        else:
            np.savez_compressed(os.path.join(save_path, filename), features=features)

    except Exception as e:
        print("[ Error while saving facial features ] - {}".format(repr(e)), file=sys.stderr)
        sys.exit(1)


def save_landmark_data(save_path, filename, landmarks):
    r"""
    Saves landmarks into a .npz file.

    Parameters
    ----------
    save_path: str
        Path in which to save landmarks.
    filename: str
        Name of the .npz file.
    landmarks: array_like
        Array containing the landmarks for each extracted frame of the video.
    """
    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.savez_compressed(os.path.join(save_path, filename), landmarks=landmarks)
    
    except Exception as e:
        print("[ Error while saving landmarks ] - {}".format(repr(e)), file=sys.stderr)
        sys.exit(1)


def save_optimizer(save_path, filename, opt):
    r"""

    Parameters
    ----------
    save_path: str
        Path in which to save the optimizer
    filename: str
        Name of the .pkl file (included extension).
    opt: Optimizer
        Optimizer object to serialize.
    """
    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filepath = os.path.join(save_path, filename)
        dump(opt, filepath)

    except Exception as e:
        print("[ Error while saving optimizer ] - {}".format(repr(e)), file=sys.stderr)
        sys.exit(1)


def unstack_eeg(eeg):
    r"""
    Rearranges EEG data from 3D arrays to 2D arrays.

    Parameters
    ----------
    eeg: array_like
        Array with shape (channels x bands, EEG features, trials).

    Returns
    -------
    eeg_data: array_like
        Array of EEG data with shape (channels x bands, EEG features x trials).
    """
    eeg_data = None
    for tr in range(40):
        if eeg_data is None:
            eeg_data = eeg[:, :, tr]
        else:
            eeg_data = np.column_stack((eeg_data, eeg[:, :, tr]))
    return eeg_data
