# © MNELAB developers
#
# License: BSD (3-clause)

import mne
import numpy as np
from scipy import stats

from mnelab.utils.dependencies import have


def find_bad_epochs_amplitude(data, amplitude_threshold):
    """Detect epochs with extreme amplitude values.

    Parameters
    ----------
    data : mne.Epochs
        Epoched data.
    amplitude_threshold : float
        Absolute amplitude threshold in Volts (V). Epochs exceeding this
        threshold (after removing the mean) will be marked as bad.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape (n_epochs,) where True indicates a bad epoch.
    """
    epochs_data = data.get_data()
    epoch_mean = epochs_data.mean(axis=-1, keepdims=True)
    bad_epochs = np.any(
        np.abs(epochs_data - epoch_mean) > amplitude_threshold, axis=(1, 2)
    )
    return bad_epochs


def find_bad_epochs_autoreject(data):
    """Detect epochs using autoreject-computed thresholds.

    Parameters
    ----------
    data : mne.Epochs
        Epoched data.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape (n_epochs,) where True indicates a bad epoch.

    Notes
    -----
    Requires the autoreject package to be installed.
    """
    if not have.get("autoreject", False):
        raise ImportError("autoreject package is required for this method")

    from autoreject import get_rejection_threshold

    reject_dict = get_rejection_threshold(data, decim=2)
    epoch_data = data.get_data()
    bad_epochs = np.zeros(len(data), dtype=bool)
    for ch_type, threshold in reject_dict.items():
        if ch_type == "eeg":
            picks = mne.pick_types(data.info, eeg=True, meg=False)
        elif ch_type == "mag":
            picks = mne.pick_types(data.info, meg="mag", eeg=False)
        elif ch_type == "grad":
            picks = mne.pick_types(data.info, meg="grad", eeg=False)
        else:
            continue

        if len(picks) == 0:
            continue

        ch_data = epoch_data[:, picks, :]
        # OR logic for bad epochs across channels
        bad_epochs |= np.any(np.abs(ch_data) > threshold, axis=(1, 2))

    return bad_epochs


def find_bad_epochs_ptp(data, ptp_threshold):
    """Detect epochs with excessive peak-to-peak amplitude.

    Parameters
    ----------
    data : mne.Epochs
        Epoched data.
    ptp_threshold : float
        Peak-to-peak amplitude threshold in Volts (V). Epochs with peak-to-peak
        amplitude exceeding this threshold will be marked as bad.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape (n_epochs,) where True indicates a bad epoch.
    """
    epoch_data = data.get_data()

    ptp_values = np.ptp(epoch_data, axis=2)
    bad_epochs = np.any(ptp_values > ptp_threshold, axis=1)

    return bad_epochs


def find_bad_epochs_kurtosis(data, kurtosis_threshold):
    """Detect epochs with abnormal kurtosis values.

    Parameters
    ----------
    data : mne.Epochs
        Epoched data.
    kurtosis_threshold : float
        Z-score threshold for kurtosis. Epochs with kurtosis z-scores exceeding
        this threshold will be marked as bad.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape (n_epochs,) where True indicates a bad epoch.
    """
    epoch_data = data.get_data()
    kurt_values = stats.kurtosis(epoch_data, axis=2, fisher=True)

    kurt_mean = np.mean(kurt_values, axis=0)
    kurt_std = np.std(kurt_values, axis=0)

    z_scores = np.abs((kurt_values - kurt_mean) / (kurt_std + 1e-10))
    bad_epochs = np.any(z_scores > kurtosis_threshold, axis=1)

    return bad_epochs
