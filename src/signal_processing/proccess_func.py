from fooof import FOOOF
from fooof.plts import plot_spectra
from fooof.utils import interpolate_spectrum
from scipy.signal import filtfilt, welch, butter, hilbert, find_peaks
from sklearn.linear_model import LinearRegression, LassoCV, Lasso
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from src.visualization import vis_cog as vc
import mne
import pandas as pd
import os
from scipy.stats import skew, kurtosis


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    :param data:
    :param lowcut:
    :param highcut:
    :param fs:
    :param order:
    :return:
    filtered data
    """

    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

def fooof_operation(data, fs):

    # Check the shape of the data
    n_channels = data.shape[1]
    # Define the window length for the analysis
    window_len = 5 # seconds
    window = window_len * fs

    #amount of windows
    n_windows = data.shape[0] // (window_len * fs)
    n_fig = n_windows//3

    # Initialize the FOOOF object
    for i in [3]:
        
        for figu in range(n_fig):
            fig, axes, = plt.subplots(2, 3, figsize=(15, 8))
            fig.suptitle(f"FOOOF Analysis for Channel {i + 1}, Figure {figu + 1}", fontsize=16)
            axes = axes.flatten()
            
            for j in range(3):
                
                # Calculate window indices
                window_idx = figu * 3 + j  # Offset by figure index
                if window_idx >= n_windows:
                    break

                start_idx = window_idx * window
                end_idx = start_idx + window

                # Compute the power spectrum of the data
                f_psd, Pxx = welch(data[start_idx:end_idx, i], fs)
                interp_ranges = [[47, 53], [97, 103]]
                freqs_int, powers_int = interpolate_spectrum(f_psd, Pxx, interp_ranges)

                # Plot the power spectrum before and after interpolation
                plot_spectra(f_psd, [Pxx, powers_int], log_powers=True,
                             labels=['Original Spectrum', 'Interpolated Spectrum'],
                             ax=axes[j])

                # Parameterize the interpolated power spectrum
                fm = FOOOF(peak_width_limits=[2, 12])
                fm.fit(freqs_int, powers_int, [3,125])  # Fit the model to the interpolated spectrum
                fm.plot(ax=axes[j + 3], add_legend=True, plot_peaks='dot')
                axes[j + 3].set_title(f"Window {window_idx + 1} - FOOOF Fit")

                # Check if alpha peak (8–13 Hz) is detected
                alpha_peak = fm.get_params('peak_params')
                alpha_in_range = any((8 <= f <= 13) for f, _, _ in alpha_peak)
                print(f"Window {window_idx + 1} - Alpha Peak Detected: {alpha_in_range}")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

def cor_data(data1, data2, labels):
    n_channels = data1.shape[1]

    # Plot the correlation between the two signals
    fig, axes, = plt.subplots(2, 2, figsize=(15, 8))
    fig.suptitle(f'Correlation between {labels[0]} and {labels[1]}', fontsize=16)

    for i in range(n_channels):
        # normalize the data z score
        data1[:, i] = (data1[:, i] - np.mean(data1[:, i])) / np.std(data1[:, i])
        data2[:, i] = (data2[:, i] - np.mean(data2[:, i])) / np.std(data2[:, i])
        # spearman correlation
        corr, _ = stats.spearmanr(data1[:, i], data2[:, i])
        print(f"Correlation between Channel {i + 1} of data1 and data2: {corr}")
        axes[i // 2, i % 2].scatter(data1[:, i], data2[:, i], s=5)
        axes[i // 2, i % 2].set_xlabel(labels[0])
        axes[i // 2, i % 2].set_ylabel(labels[1])
        axes[i // 2, i % 2].set_title(f"Channel {i + 1} - Correlation: {corr:.2f}")

    plt.tight_layout()
    plt.show()

def barin_process(obj, status=2, period=1, window_length=3):
    # Extract data
    data = obj.extract_trials(status, period)

    # Extract the upper electrodes
    upper_channels = data[0][:, 12:16] # Extract the upper electrodes



    # Plot the full data
    vc.plot_trail(data)

    if status == 2:
        # limit the data to the alpha calibration
        alpha_data_start = np.where(data[2] == 21)[0][0]
        alpha_data_end = np.where(data[2] == 23)[0][0]
        triggers = data[2][alpha_data_start:alpha_data_end + 1]
        triggers_time = data[3][alpha_data_start:alpha_data_end + 1]
        alpha_data_start = data[3][alpha_data_start]
        alpha_data_end = data[3][alpha_data_end]
        mask_data = (data[1] >= alpha_data_start) & (data[1] <= alpha_data_end)
        alpha_data = upper_channels[mask_data, :]
        time_stamps = data[1][mask_data]
        start_time = min(time_stamps[0], triggers_time[0])
        time_stamps -= start_time
        triggers_time -= start_time
    else:
        alpha_data = upper_channels
        time_stamps = data[1]
        triggers = data[2]
        triggers_time = data[3]

    # Perform FOOOF on Calibration data
    # fooof_operation(alpha_data, 250)

    # Create spectrogram of filtered_cal
    # vc.plot_spectrogram(alpha_data, 250, 'Spectrogram of filtered calibration data')

    filtered_cal_alpha = bandpass_filter(alpha_data, 8, 13, 250, 4)
    filtered_cal_EMG = bandpass_filter(alpha_data, 35, 124, 250, 4)
    filtered_cal_beta = bandpass_filter(alpha_data, 12, 30, 250, 4)

    cor_data(filtered_cal_alpha, filtered_cal_EMG, ['Alpha', 'EMG'])
    cor_data(filtered_cal_beta, filtered_cal_EMG, ['Beta', 'EMG'])
    cor_data(filtered_cal_alpha, filtered_cal_beta, ['Alpha', 'Beta'])

    vc.create_plot(alpha_data, time_stamps, triggers_time, triggers, title='EEG band filtered signal')

    if status == 2:
        upper_channels = alpha_data

    # Sample frequency
    fs = obj.fs

    # Define the frequency bands
    alpha_band = (8, 13)
    emg_band = (35, 124)
    beta_band = (12, 30)

    # Normalize by spectral width
    alpha_width = alpha_band[1] - alpha_band[0]
    emg_width = emg_band[1] - emg_band[0]
    beta_width = beta_band[1] - beta_band[0]

    n_samples, n_channels = upper_channels.shape
    window_size = int(window_length * fs)
    n_windows = n_samples // window_size

    alpha_power = np.zeros((n_windows, n_channels))
    emg_power = np.zeros((n_windows, n_channels))
    beta_power = np.zeros((n_windows, n_channels))

    # Iterate through windows
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        segment = upper_channels[start:end, :]

        for ch in range(n_channels):
            freqs, psd = welch(segment[:, ch], fs, nperseg=window_size)

            # Calculate power in alpha band
            alpha_idx = np.logical_and(freqs >= alpha_band[0], freqs <= alpha_band[1])
            alpha_power[i, ch] = np.sum(psd[alpha_idx])/alpha_width

            # Calculate power in EMG band
            emg_idx = np.logical_and(freqs >= emg_band[0], freqs <= emg_band[1])
            emg_power[i, ch] = np.sum(psd[emg_idx])/emg_width

            # Calculate power in beta band
            beta_idx = np.logical_and(freqs >= beta_band[0], freqs <= beta_band[1])
            beta_power[i, ch] = np.sum(psd[beta_idx])/beta_width

    # Define a threshold for outliers (e.g., 3 standard deviations from the mean)
    threshold = 3

    # Initialize masks for all channels
    outlier_mask = np.zeros_like(emg_power, dtype=bool)

    # Create clean versions of the power arrays by masking out outliers
    alpha_power_clean = np.zeros_like(alpha_power)
    emg_power_clean = np.zeros_like(emg_power)

    # Identify outliers for each channel independently
    for ch in range(emg_power.shape[1]):  # Iterate over channels
        mean_emg = np.mean(emg_power[:, ch])
        std_emg = np.std(emg_power[:, ch])
        outlier_mask[:, ch] = (emg_power[:, ch] > (mean_emg + threshold * std_emg)) | \
                              (emg_power[:, ch] < (mean_emg - threshold * std_emg))

        # Apply mask for each channel
        alpha_power_clean[:, ch] = np.where(~outlier_mask[:, ch], alpha_power[:, ch], np.nan)
        emg_power_clean[:, ch] = np.where(~outlier_mask[:, ch], emg_power[:, ch], np.nan)

    # Drop rows with NaN (outliers) for each channel separately
    alpha_power_clean = alpha_power_clean[~np.isnan(alpha_power_clean).any(axis=1)]
    emg_power_clean = emg_power_clean[~np.isnan(emg_power_clean).any(axis=1)]


    # Calculate R^2 for each channel no outliers
    r_squared_values = np.zeros(n_channels)
    for ch in range(n_channels):
        # Linear regression
        model = LinearRegression()
        X = emg_power_clean[:, ch].reshape(-1, 1)  # Independent variable
        y = alpha_power_clean[:, ch]  # Dependent variable
        model.fit(X, y)
        r_squared_values[ch] = model.score(X, y)  # R^2

    # Plot the correlation between EMG and Alpha Power no outliers
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2x2 subplot grid
    fig.suptitle('Correlation between EMG and Alpha Power-no outliers', fontsize=16)

    for idx, ax in enumerate(axes.flatten()):
        channel = idx
        ax.scatter(emg_power_clean[:, channel], alpha_power_clean[:, channel], alpha=0.6)
        ax.set_title(f'Channel {channel+13}- R^2: {r_squared_values[channel]:.2f}')
        ax.set_xlabel('EMG Power')
        ax.set_ylabel('Alpha Power')

    plt.tight_layout()
    plt.show()

    # Calculate R^2 for each channel
    r_squared_values = np.zeros(n_channels)
    for ch in range(n_channels):
        # Linear regression
        model = LinearRegression()
        X = emg_power[:, ch].reshape(-1, 1)  # Independent variable
        y = alpha_power[:, ch]  # Dependent variable
        model.fit(X, y)
        r_squared_values[ch] = model.score(X, y)  # R^2

    # Plot the correlation between EMG and Alpha Power
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2x2 subplot grid
    fig.suptitle('Correlation between EMG and Alpha Power', fontsize=16)

    for idx, ax in enumerate(axes.flatten()):
        channel = idx
        ax.scatter(emg_power[:, channel], alpha_power[:, channel], alpha=0.6)
        ax.set_title(f'Channel {channel+13}- R^2: {r_squared_values[channel]:.2f}')
        ax.set_xlabel('EMG Power')
        ax.set_ylabel('Alpha Power')

    plt.tight_layout()
    plt.show()

def calc_rms(data, timestamps, window_length=3, overlap=0.5):
    """
    Calculate RMS using actual timestamps instead of assuming fs.

    Parameters:
        data (ndarray): Input data to calculate RMS. Shape (n_samples, n_channels).
        timestamps (ndarray): Actual timestamps for each sample. Shape (n_samples,).
        window_length (float): Window length in seconds.
        overlap (float): Overlap between consecutive windows (0 to 1).

    Returns:
        rms (ndarray): RMS values for each channel. Shape (n_windows, n_channels).
        time_vec (ndarray): Time vector based on actual timestamps.
    """
    # Calculate window size and step in samples
    fs_estimated = 1 / np.mean(np.diff(timestamps))  # Estimate fs from timestamps
    window_size = int(window_length * fs_estimated)
    window_step = int(window_size * (1 - overlap))

    # Calculate number of windows
    n_windows = (data.shape[0] - window_size) // window_step + 1

    # Initialize RMS array
    rms = np.zeros((n_windows, data.shape[1]))
    time_vec = []

    # Calculate RMS for each window
    for i in range(n_windows):
        start = i * window_step
        end = start + window_size
        window = data[start:end, :]
        rms[i, :] = np.sqrt(np.mean(window ** 2, axis=0))

        # Append the center time of the window
        time_vec.append(np.mean(timestamps[start:end]))

    # Convert time_vec to ndarray
    time_vec = np.array(time_vec)

    return rms, time_vec

def extract_rms_by_band(processor, overlap=0.5):
    """
    Process experiments to calculate RMS for multiple frequency bands.

    Parameters:
        processor: An object with `play_periods` and `extract_trials` methods.
        overlap (float): Overlap fraction between consecutive windows.

    Returns:
        rms_results (dict): Dictionary with keys as frequency bands and values
                            as lists of RMS arrays for each experiment.
                            Example: {'alpha': [rms_exp1, rms_exp2, ...]}.
    """
    # Define the frequency bands
    frequency_bands = {
        'alpha': (8, 13),
        'beta': (12, 30),
        'emg': (35, 124)
    }

    fs = processor.fs

    # Calculate optimal window sizes for each band
    def calculate_window_size(low_freq, cycles=2):
        period = 1 / low_freq
        return cycles * period * fs  # Convert to samples

    window_sizes = {band: int(calculate_window_size(lowcut)) for band, (lowcut, _) in frequency_bands.items()}

    # Initialize results dictionary
    rms_results = {band: [] for band in frequency_bands}

    # Loop through each experiment
    for period in range(len(processor.play_periods)):
        # Extract the data for the current experiment
        data = processor.extract_trials(status=1, period=period)  # Assuming status = 1
        raw_data = data[0]  # Extract the acquired samples (n_samples x k_channels)

        # Extract channels 13–16 for EEG-related frequency bands
        eeg_channels = raw_data[:, 12:16]  # Channels 13–16 (0-based indexing)

        # Loop through each frequency band
        for band, (lowcut, highcut) in frequency_bands.items():
            # Apply bandpass filter
            if band in ['alpha', 'beta']:
                filtered_data = bandpass_filter(eeg_channels, lowcut, highcut, fs, order=4)
            else:
                filtered_data = bandpass_filter(raw_data, lowcut, highcut, fs, order=4)

            # Convert window size to seconds
            window_length = window_sizes[band] / fs

            # Calculate RMS for the filtered data
            rms, time_vec = calc_rms(filtered_data, data[1], window_length=window_length, overlap=overlap)

            # Store the RMS result
            rms_results[band].append({'experiment': period, 'rms': rms, 'time_vec': time_vec})

    return rms_results

def plot_alpha_calibration(processor, cycles=2, overlap=0.5):
    """
    Extract and plot the alpha calibration signal with RMS.

    Parameters:
        processor: An object with `extract_trials` method and `fs` attribute.
        cycles (int): Number of cycles for the window length.
        overlap (float): Overlap fraction for RMS calculation.

    Returns:
        None
    """
    # Extract the data for the current experiment
    data = processor.extract_trials(status=2)

    fs = processor.fs

    # Extract alpha calibration based on triggers 21 and 23
    alpha_data_start = np.where(data[2] == 21)[0][0]
    alpha_data_end = np.where(data[2] == 23)[0][0]
    alpha_start_time = data[3][alpha_data_start]
    alpha_end_time = data[3][alpha_data_end]

    # Mask to extract signal and timestamps within the calibration period
    mask_data = (data[1] >= alpha_start_time) & (data[1] <= alpha_end_time)
    alpha_signal = data[0][mask_data, 12:16]  # Last four channels
    time_stamps = data[1][mask_data]

    # Adjust timestamps to start at 0
    time_stamps -= time_stamps[0]

    # Filter signal for alpha frequencies (8–13 Hz)
    filtered_signal = bandpass_filter(alpha_signal, lowcut=8, highcut=13, fs=fs, order=4)
    EEG_signal = bandpass_filter(alpha_signal, lowcut=0.5, highcut=50, fs=fs, order=4)

    period = 1 / 8  # Period of the alpha frequency
    window_length = cycles * period

    # Calculate RMS
    rms, time_vec = calc_rms(filtered_signal, data[1], window_length=window_length, overlap=overlap)

    time_vec -= time_vec[0]  # Adjust time vector to start at 0

    # Normalize filtered signal and RMS to [0, 1]
    normalized_signal = (EEG_signal - np.min(EEG_signal)) / (
                np.max(EEG_signal) - np.min(EEG_signal))
    normalized_signal -= np.mean(normalized_signal)
    normalized_rms = (rms - np.min(rms)) / (np.max(rms) - np.min(rms))

    # Create MNE Raw object
    ch_names = ["13", "14", "15", "16"]
    info = mne.create_info(ch_names, sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(alpha_signal.T, info)  # MNE expects (n_channels x n_samples)

    # Bandpass filter the signal for alpha range
    raw_filtered = raw.copy().filter(l_freq=8, h_freq=13, fir_design='firwin')
    filtered_signal = raw_filtered.get_data().T  # Get filtered data (n_samples x n_channels)

    # Compute the alpha envelope using the Hilbert transform
    alpha_envelope = np.abs(hilbert(filtered_signal, axis=0))

    # Set the threshold based on the alpha envelope
    threshold = np.percentile(alpha_envelope, 60)

    # Plot the filtered signal and RMS
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Alpha Calibration: Filtered Signal and RMS", fontsize=16)

    for i, ax in enumerate(axes.flatten()):
        ax.plot(time_stamps, normalized_signal[:, i], label="Filtered Signal", alpha=0.8)
        ax.plot(time_vec, normalized_rms[:, i], label="RMS", linestyle="--", alpha=0.8)
        ax.fill_between(
            time_stamps,
            normalized_signal[:, i],
            where=alpha_envelope[:, i] > threshold,
            color='orange',
            alpha=0.5,
            label="Alpha Detected"
        )
        ax.set_title(f"Channel {i + 13}")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")

    axes[-1, -1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()
