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
import vis_cog as vc
import mne
import pandas as pd
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

    # cor_data(filtered_cal_alpha, filtered_cal_EMG, ['Alpha', 'EMG'])
    # cor_data(filtered_cal_beta, filtered_cal_EMG, ['Beta', 'EMG'])
    # cor_data(filtered_cal_alpha, filtered_cal_beta, ['Alpha', 'Beta'])
    #
    # vc.create_plot(alpha_data, time_stamps, triggers_time, triggers, title='EEG band filtered signal')

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

def extract_rms_features(rms_data, time_vec, features, band, channels, fs):
    for idx, ch in enumerate(channels):
        channel_data = rms_data[:, idx] # RMS data for the current channel
        t_total = time_vec[-1] - time_vec[0]    # Total duration in seconds

        # Statistical Features
        features[f"{band}_channel{ch + 1}_mean_rms"] = np.mean(channel_data)
        features[f"{band}_channel{ch + 1}_std_rms"] = np.std(channel_data)
        features[f"{band}_channel{ch + 1}_max_rms"] = np.max(channel_data)
        features[f"{band}_channel{ch + 1}_min_rms"] = np.min(channel_data)
        features[f"{band}_channel{ch + 1}_iqr_rms"] = np.percentile(channel_data, 75) - np.percentile(channel_data, 25)
        features[f"{band}_channel{ch + 1}_entropy_rms"] = -np.sum(channel_data * np.log2(channel_data + 1e-12))

        # Temporal Features
        rms_gradient = np.gradient(channel_data, time_vec)
        features[f"{band}_channel{ch + 1}_rms_slope"] = np.mean(rms_gradient)
        features[f"{band}_channel{ch + 1}_rms_gradient_variance"] = np.var(rms_gradient)
        features[f"{band}_channel{ch + 1}_max_gradient"] = np.max(rms_gradient)
        features[f"{band}_channel{ch + 1}_min_gradient"] = np.min(rms_gradient)
        features[f"{band}_channel{ch + 1}_gradient_range"] = np.max(rms_gradient) - np.min(rms_gradient)
        features[f"{band}_channel{ch + 1}_positive_gradient_count"] = np.sum(rms_gradient > 0)
        features[f"{band}_channel{ch + 1}_negative_gradient_count"] = np.sum(rms_gradient < 0)
        features[f"{band}_channel{ch + 1}_positive_gradient_ratio"] = np.sum(rms_gradient > 0) / len(rms_gradient)
        features[f"{band}_channel{ch + 1}_negative_gradient_ratio"] = np.sum(rms_gradient < 0) / len(rms_gradient)

        # Event-Based Features
        threshold = np.mean(channel_data) + 2 * np.std(channel_data)
        peaks = [r for r in channel_data if r > threshold]
        features[f"{band}_channel{ch + 1}_peaks_count_per_sec"] = len(peaks) / t_total
        features[f"{band}_channel{ch + 1}_mean_peak_height"] = np.mean(peaks) if peaks else 0

        # Dynamic Features
        high_rms_duration_sec  = np.sum(channel_data > np.mean(channel_data) + np.std(channel_data))/len(channel_data)
        features[f"{band}_channel{ch + 1}_high_rms_duration"] = high_rms_duration_sec
        features[f"{band}_channel{ch + 1}_activity_entropy"] = -np.sum(
            (channel_data / np.sum(channel_data)) * np.log2((channel_data / np.sum(channel_data)) + 1e-12)
        )

    return features

def extract_frequency_features(raw_data, features, band, channels, fs, window_length=3, overlap=0.5):
    """
    Extract frequency-domain features from the signal using windowed PSD calculations.

    Parameters:
        raw_data (ndarray): Raw signal data, shape (n_samples, n_channels).
        features (dict): Dictionary to store the extracted features.
        band (str): Frequency band being processed (e.g., 'alpha', 'beta', 'emg').
        channels (iterable): Channels to process.
        fs (int): Sampling frequency of the signal.
        window_length (float): Window length in seconds for PSD.
        overlap (float): Overlap fraction for consecutive windows.

    Returns:
        features (dict): Updated dictionary with frequency-domain features.
    """
    # Band definitions for filtering
    band_ranges = {
        'alpha': (8, 13),
        'beta': (12, 30),
        'emg': (35, 124)
    }
    lowcut, highcut = band_ranges[band]

    # Calculate window size and step in samples
    window_size = int(window_length * fs)
    window_step = int(window_size * (1 - overlap))

    for ch in channels:
        channel_data = raw_data[:, ch]

        # Split signal into overlapping windows
        n_windows = (len(channel_data) - window_size) // window_step + 1
        windowed_psd = []

        for i in range(n_windows):
            start = i * window_step
            end = start + window_size
            window = channel_data[start:end]

            # Calculate Welch's PSD for the current window
            freqs, psd = welch(window, fs=fs, nperseg=window_size)

            # Restrict PSD to the band of interest
            band_mask = (freqs >= lowcut) & (freqs <= highcut)
            band_psd = psd[band_mask]
            windowed_psd.append(np.sum(band_psd))  # Sum power in the band

        # Convert windowed PSD to numpy array for aggregation
        windowed_psd = np.array(windowed_psd)

        # Calculate frequency-domain features for the band
        features[f"{band}_channel{ch + 1}_mean_band_power"] = np.mean(windowed_psd)
        features[f"{band}_channel{ch + 1}_band_power_variance"] = np.var(windowed_psd)
        features[f"{band}_channel{ch + 1}_max_band_power"] = np.max(windowed_psd)
        features[f"{band}_channel{ch + 1}_min_band_power"] = np.min(windowed_psd)
        features[f"{band}_channel{ch + 1}_band_power_iqr"] = np.percentile(windowed_psd, 75) - np.percentile(
            windowed_psd, 25)

        # Temporal dynamics within the band
        features[f"{band}_channel{ch + 1}_early_band_power"] = np.mean(windowed_psd[:n_windows // 3])
        features[f"{band}_channel{ch + 1}_late_band_power"] = np.mean(windowed_psd[-n_windows // 3:])

        # Complexity metrics
        features[f"{band}_channel{ch + 1}_spectral_entropy"] = -np.sum(
            (windowed_psd / np.sum(windowed_psd + 1e-12)) * np.log2(windowed_psd / np.sum(windowed_psd + 1e-12) + 1e-12)
        )

    return features

def extract_time_features(raw_data, timestamps, features, band, channels, fs, threshold_factor=2,
    window_length=3.0, overlap=0.5):
    """
    Extract time-domain features in short windows, then aggregate them into single values.

    Parameters
    ----------
    raw_data : ndarray
        Signal data, shape (n_samples, n_channels). This can be raw or bandpass-filtered.
    timestamps : ndarray
        Timestamps corresponding to the signal samples, shape (n_samples,).
    features : dict
        Dictionary to store the aggregated features.
    band : str
        Frequency band name (e.g., 'alpha', 'beta', 'emg').
    channels : iterable
        Which channel indices to process.
    fs : int or float
        Sampling frequency in Hz.
    threshold_factor : float
        Multiplier for threshold-based peak detection within each window.
    window_length : float
        Length in seconds of each analysis window.
    overlap : float
        Overlap fraction (0.0 to 1.0) for consecutive windows.

    Returns
    -------
    features : dict
        Updated dictionary with aggregated time-domain features for each channel and band.
    """
    # Decide which channels to use based on band
    if band in ['alpha', 'beta']:
        # Channels 13-16 in 1-based => indices 12..15 in 0-based
        forehead_set = {12, 13, 14, 15}
        relevant_channels = sorted(forehead_set.intersection(channels))
    else:
        # e.g. 'emg' => use all channels (or whatever the user provided)
        relevant_channels = sorted(channels)

    # Basic parameters
    n_samples = raw_data.shape[0]
    total_duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0

    # Calculate number of samples per window and step
    window_size = int(window_length * fs)
    step_size = int(window_size * (1 - overlap))
    if step_size < 1:
        step_size = 1  # Prevent zero or negative step in extreme overlap cases
    n_windows = max(0, (n_samples - window_size) // step_size + 1)

    # Define a helper function for easier aggregator naming
    def add_stat(name_prefix, array, ch):
        """
        For a given 'array' of window-level values, compute various statistics
        (mean, std, median, min, max, p10, p90) and add them to 'features' dict
        under a consistent naming scheme.

        name_prefix might be something like "absmean" or "peakcount".
        """
        if len(array) == 0:
            # If no windows, just store 0
            features[f"{band}_channel{ch}_wnd_{name_prefix}_mean"] = 0
            features[f"{band}_channel{ch}_wnd_{name_prefix}_std"] = 0
            features[f"{band}_channel{ch}_wnd_{name_prefix}_median"] = 0
            features[f"{band}_channel{ch}_wnd_{name_prefix}_min"] = 0
            features[f"{band}_channel{ch}_wnd_{name_prefix}_max"] = 0
            features[f"{band}_channel{ch}_wnd_{name_prefix}_p10"] = 0
            features[f"{band}_channel{ch}_wnd_{name_prefix}_p90"] = 0
            return

        features[f"{band}_channel{ch}_wnd_{name_prefix}_mean"] = np.mean(array)
        features[f"{band}_channel{ch}_wnd_{name_prefix}_std"] = np.std(array)
        features[f"{band}_channel{ch}_wnd_{name_prefix}_median"] = np.median(array)
        features[f"{band}_channel{ch}_wnd_{name_prefix}_min"] = np.min(array)
        features[f"{band}_channel{ch}_wnd_{name_prefix}_max"] = np.max(array)
        features[f"{band}_channel{ch}_wnd_{name_prefix}_p10"] = np.percentile(array, 10)
        features[f"{band}_channel{ch}_wnd_{name_prefix}_p90"] = np.percentile(array, 90)

    # Loop over channels
    for ch in relevant_channels:
        # Prepare lists to store the per-window calculations:
        window_absmeans = []
        window_variances = []
        window_skewnesses = []
        window_kurtoses = []
        window_amplranges = []
        window_peakcounts = []
        window_peakampls = []
        window_zcrs = []  # zero-crossing rates

        # Slide over the signal in windows
        channel_data = raw_data[:, ch]
        for w_idx in range(n_windows):
            start = w_idx * step_size
            end = start + window_size
            segment = channel_data[start:end]

            if len(segment) == 0:
                continue  # skip empty segments

            # 1. Basic amplitude-based stats
            absmean = np.mean(np.abs(segment))  # absolute mean
            var_ = np.var(segment)
            skew_ = skew(segment)
            kurt_ = kurtosis(segment)
            amp_range = np.max(segment) - np.min(segment)

            # 2. Peak detection in this window
            local_thresh = np.mean(segment) + threshold_factor * np.std(segment)
            peaks, _ = find_peaks(segment, height=local_thresh)
            peak_count = len(peaks)
            peak_ampl = np.mean(segment[peaks]) if peak_count > 0 else 0

            # 3. Zero-crossing rate in the window
            zc_rate = 0
            if len(segment) > 1:
                zero_crossings = np.sum(
                    np.diff(np.sign(segment - np.mean(segment))) != 0
                )
                zc_rate = zero_crossings / (len(segment) - 1)  # or len(segment)

            # Store the window-level results
            window_absmeans.append(absmean)
            window_variances.append(var_)
            window_skewnesses.append(skew_)
            window_kurtoses.append(kurt_)
            window_amplranges.append(amp_range)
            window_peakcounts.append(peak_count)
            window_peakampls.append(peak_ampl)
            window_zcrs.append(zc_rate)

        # ---------------------------------------------------------------------
        # Aggregate these window-level values into single (or multiple) scalars
        # ---------------------------------------------------------------------

        ch_name = ch + 1  # for 1-based labeling in feature names

        # Aggregator stats: mean, std, median, min, max, p10, p90
        add_stat("absmean", window_absmeans, ch_name)
        add_stat("variance", window_variances, ch_name)
        add_stat("skewness", window_skewnesses, ch_name)
        add_stat("kurtosis", window_kurtoses, ch_name)
        add_stat("amplrange", window_amplranges, ch_name)
        add_stat("peakcount", window_peakcounts, ch_name)
        add_stat("peakampl", window_peakampls, ch_name)
        add_stat("zcr", window_zcrs, ch_name)

    return features

def create_feature_table(processor, data, overlap=0.5):
    """
    Create a feature table with extracted features for each period.

    Parameters:
        processor: Object with `play_periods` and `extract_trials` methods.
        data: Object containing metadata (e.g., difficulty ratings in `trail_cog_nasa`).
        overlap (float): Overlap fraction for RMS computation.

    Returns:
        feature_table (pd.DataFrame): Feature table with features for each period.
    """
    # Define the frequency bands
    frequency_bands = {
        'alpha': (8, 13),
        'beta': (12, 30),
        'emg': (35, 124)
    }

    # Initialize feature table
    feature_rows = []

    # Compute RMS for each frequency band
    rms_results = extract_rms_by_band(processor, overlap=overlap)

    # Process each period
    for period_idx, period in enumerate(processor.play_periods):
        # Extract data for the current period
        trial_data = processor.extract_trials(status=1, period=period_idx)
        raw_data = trial_data[0]  # Signal data (n_samples x n_channels)
        timestamps = trial_data[1]  # Timestamps for the signal

        # Identify if this is a calibration period
        is_calibration = period_idx < 10
        period_type = "Calibration" if is_calibration else "Task"
        difficulty_rating = data.trail_cog_nasa[period_idx]  # Subject's difficulty rating

        # Initialize features dictionary for the period
        features = {
            "Period": period_idx + 1,
            "Type": period_type,
            "Difficulty Rating": difficulty_rating,
        }

        # Iterate through each frequency band
        for band, band_results in frequency_bands.items():
            # Select the appropriate channels for the band
            if band in ['alpha', 'beta']:
                channels = range(12, 16)  # EEG channels (13-16 in 1-based indexing)
            else:
                channels = range(raw_data.shape[1])  # All channels for EMG

            # Extract band-specific RMS data and timestamps
            rms_data = rms_results[band][period_idx]["rms"]
            time_vec = rms_results[band][period_idx]["time_vec"]

            # Call the feature extraction functions
            features = extract_rms_features(rms_data, time_vec, features, band, channels, processor.fs)
            features = extract_frequency_features(raw_data, features, band, channels, processor.fs)
            filtered_data = bandpass_filter(raw_data, band_results[0], band_results[1], processor.fs, order=4)
            features = extract_time_features(filtered_data, timestamps, features, band, channels, processor.fs)

        # Append the features for the current period
        feature_rows.append(features)

    # Convert to DataFrame
    feature_table = pd.DataFrame(feature_rows)

    return feature_table

def filter_highly_correlated_features(df, label_col="Difficulty Rating", corr_threshold=0.85):
    """
    Removes one feature from each pair of highly correlated features (>= corr_threshold),
    keeping the one that has higher correlation with the label.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame that includes the label_col and numeric feature columns.
    label_col : str
        The name of the label column to preserve in the data.
    corr_threshold : float
        Threshold above which two features are considered "highly correlated."

    Returns
    -------
    df_filtered : pd.DataFrame
        A DataFrame with the label_col plus the remaining (non-duplicate) features.
    """
    # 1. Separate out the label column (ensure it's in the DataFrame).
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame.")

    # We'll keep the label aside, but preserve it in the final output.
    label_series = df[label_col]

    # 2. Drop non-numeric or irrelevant columns if needed, or just select numeric columns.
    #    Make sure to exclude the label from the correlation matrix below.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)  # don't want label in the correlation among features

    # Create a sub-DataFrame of only numeric feature columns
    df_features = df[numeric_cols]

    df_features = df_features.drop(columns="Period", errors="ignore")

    # 3. Compute the correlation matrix among the features
    corr_matrix = df_features.corr().abs()  # absolute correlation

    # 4. Sort features by their absolute correlation with the label (descending).
    #    We'll use this to decide which feature to keep if two features are correlated.
    label_corr = df_features.corrwith(label_series).abs().sort_values(ascending=False)

    # We'll track which features we *remove* in a set
    to_remove = set()

    # 5. For each pair above threshold, remove one feature
    #    (We can do an upper triangle scan to avoid double counting).
    #    We'll keep the feature that has the *higher* label correlation.
    for i in range(len(label_corr)):
        feature_i = label_corr.index[i]
        if feature_i in to_remove:
            # Already removed
            continue

        for j in range(i + 1, len(label_corr)):
            feature_j = label_corr.index[j]
            if feature_j in to_remove:
                continue

            # Check correlation between feature_i and feature_j
            if corr_matrix.loc[feature_i, feature_j] >= corr_threshold:
                # They are highly correlated.
                # Remove the one with the lower label correlation => keep the bigger label-corr.
                if label_corr[feature_i] >= label_corr[feature_j]:
                    # remove j
                    to_remove.add(feature_j)
                else:
                    # remove i
                    to_remove.add(feature_i)
                    break  # feature_i is removed, no need to compare it further

    # 6. Construct the final filtered DataFrame
    remaining_features = [f for f in numeric_cols if f not in to_remove]

    # Re-create a DataFrame with the remaining features plus the label
    df_filtered = pd.concat([df[remaining_features], label_series], axis=1)

    return df_filtered

def get_top_correlated_features(df, label_col="Difficulty Rating", top_n=20, plot_corr=False, save_path=None):
    """
    Return the top_n features most correlated with the label_col in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing all the features plus the label column.
    label_col : str, optional
        Name of the column that holds the target label/difficulty rating.
    top_n : int, optional
        How many of the most correlated features to return.
    plot_corr : bool, optional
        Whether to plot the correlation between the top features and the label.
    save_path : str, optional
        Path to save the plots if plot_corr is True.

    Returns
    -------
    top_feature_names : list of str
        Names of the features, sorted by descending correlation magnitude.
    top_correlations : ndarray
        The absolute correlation values corresponding to those features,
        in the same order.
    """
    # Filter highly correlated features
    df_filtered = filter_highly_correlated_features(df)

    # Drop non-numeric columns
    df_numeric = df_filtered.drop(columns=["Period"], errors="ignore")

    # 1. Compute the correlation matrix
    corr_matrix = df_numeric.corr()

    # 2. Get the series of correlations with the target label
    #    We take the absolute value so we can rank by magnitude (whether +/-).
    if label_col not in corr_matrix.columns:
        raise ValueError(f"Label column '{label_col}' not in DataFrame or not numeric.")

    label_corr = corr_matrix[label_col].abs()

    # 3. Remove the label_col itself from the series (so we don't get correlation of label w/ label)
    label_corr = label_corr.drop(label_col, errors='ignore')

    # 4. Sort by descending correlation magnitude
    label_corr_sorted = label_corr.sort_values(ascending=False)

    # 5. Take the top_n features
    top_features = label_corr_sorted.head(top_n)

    # 6. Extract the feature names and correlation values
    top_feature_names = top_features.index.tolist()
    top_correlations = top_features.values

    if plot_corr:
        # Plot the top_n features
        y1 = df[label_col]
        x = range(len(y1))
        for feature in top_feature_names:
            y2 = df[feature]
            # Normalize the feature values to the same scale as the label (min max scaling)
            y2 = (y2 - y2.min()) / (y2.max() - y2.min()) * (y1.max() - y1.min()) + y1.min()
            plt.figure(figsize=(8, 6))
            # Difficulties (y1)
            plt.scatter(x, y1, alpha=0.6, color='red', label=f"{label_col} (red)")
            plt.plot(x, y1, color='red', alpha=0.6, label='_nolegend_')

            # Feature (y2)
            plt.scatter(x, y2, alpha=0.6, color='blue', label=feature)
            plt.plot(x, y2, color='blue', alpha=0.6, label='_nolegend_')

            # Title, labels
            plt.title(f"{feature} vs. {label_col} (correlation: {label_corr[feature]:.2f})")
            plt.xlabel("maze number")
            plt.ylabel("Value - Normalized")

            # Show legend for only the scatter points
            plt.legend(loc='upper right')

            plt.show(block=False)
            # Save the plot if save_path is provided
            if save_path is not None:
                plt.savefig(save_path[:-3] + f"{feature}_vs_{label_col}.png")
                # Close the plot
                plt.close()

    return top_feature_names, top_correlations

def lasso_feature_selection(df, candidate_features, label_col="Difficulty Rating", n_splits=5,
    random_state=42, alphas=None, use_loo=False):
    """
    Selects features using Lasso with cross-validation.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing both the candidate features and the label_col.
    candidate_features : list of str
        List of feature column names (e.g., your top-20 correlated features).
    label_col : str
        The column name of the target/label (numerical) in `df`.
    n_splits : int, optional
        Number of folds in KFold cross-validation (ignored if use_loo=True).
    random_state : int, optional
        Random seed for reproducibility in KFold.
    alphas : array-like, optional
        List of alpha values for Lasso to try.
        If None, a default log-spaced range is used.
    use_loo : bool, optional
        If True, use Leave-One-Out CV instead of KFold.

    Returns
    -------
    selected_features : list of str
        Names of the features that have non-zero coefficients in the final Lasso model.
    best_alpha : float
        The alpha chosen by LassoCV.
    lasso_model : LassoCV
        The fitted LassoCV model (you can inspect coefficients, etc.).
    """

    # 1. Subset DataFrame to candidate features + label
    #    We'll drop rows with NaN if necessary, or you can handle them differently.
    df_sub = df[candidate_features + [label_col]].dropna()

    # Separate X and y
    X = df_sub[candidate_features].values
    y = df_sub[label_col].values

    # normalize X to have zero mean and unit variance (important for Lasso)
    X = StandardScaler().fit_transform(X)

    # 2. Define cross-validation approach
    if use_loo:
        cv = LeaveOneOut()
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # 3. Define alpha grid if not provided
    if alphas is None:
        # For small data, you can try a modest range.
        # Or you can broaden it if you want more exhaustive search.
        alphas = np.logspace(-3, 1, 50)  # from 0.001 to 10, in log scale

    # 4. Fit LassoCV
    lasso_cv = LassoCV(
        alphas=alphas,
        cv=cv,
        max_iter=10000,
        random_state=random_state
    )
    lasso_cv.fit(X, y)

    # 5. Retrieve the best alpha
    best_alpha = lasso_cv.alpha_

    # 6. Identify which features are non-zero
    coefs = lasso_cv.coef_
    selected_features = [
        feat for feat, coef in zip(candidate_features, coefs)
        if abs(coef) > 1e-10
    ]

    return selected_features, best_alpha, lasso_cv

def build_and_evaluate_model(
    df,
    candidate_features,
    label_col="Difficulty Rating",
    use_loo=False,
    n_splits=5,
    random_state=42,
    alphas=None,
    retrain_lasso=True
):
    """
    1) Select features via Lasso-based feature selection.
    2) Build a model using only those selected features.
    3) Evaluate the model with cross-validation and return some performance metrics.

    Parameters
    ----------
    df : pd.DataFrame
        The full DataFrame containing all potential features + the label.
    candidate_features : list of str
        List of columns to consider in Lasso feature selection (e.g., your top 20).
    label_col : str, default="Difficulty Rating"
        The name of the target column.
    use_loo : bool, default=False
        If True, use Leave-One-Out CV inside Lasso. Otherwise uses KFold(n_splits).
    n_splits : int, default=5
        Number of splits for KFold (ignored if use_loo=True).
    random_state : int, default=42
        For reproducibility in KFold shuffling.
    alphas : array-like, optional
        List of alpha values for Lasso to try. If None, default logspace is used.
    retrain_lasso : bool, default=False
        If True, retrain a Lasso model with the best alpha.
        If False, we use a simple LinearRegression on the selected features.

    Returns
    -------
    model : estimator
        The final trained model (either Lasso or LinearRegression).
    selected_features : list of str
        The subset of features that were selected by Lasso.
    performance_dict : dict
        A dictionary containing mean/stdev of R^2 and MSE across CV.
    """

    # 1) Call your Lasso-based selection function
    selected_features, best_alpha, lasso_cv_model = lasso_feature_selection(
        df=df,
        candidate_features=candidate_features,
        label_col=label_col,
        n_splits=n_splits,
        random_state=random_state,
        alphas=alphas,
        use_loo=use_loo
    )

    # 2) Subset the DataFrame to the selected features + label
    df_sub = df[selected_features + [label_col]].dropna()
    X_sub = df_sub[selected_features].values
    # normalize X to have zero mean and unit variance (important for Lasso)
    X_sub = StandardScaler().fit_transform(X_sub)
    y_sub = df_sub[label_col].values

    pca = PCA()
    pca.fit(X_sub)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.show()

    # take the first 4 components
    pca = PCA(n_components=4)
    X_sub = pca.fit_transform(X_sub)


    # 3) Build final model
    if retrain_lasso:
        # We'll retrain Lasso using the best alpha found
        final_model = Lasso(alpha=best_alpha, max_iter=10000, random_state=random_state)
    else:
        # We'll build a simple linear regression model
        final_model = LinearRegression()

    # 4) Evaluate with cross-validation on these selected features
    #    We'll compute R^2 and MSE for demonstration
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # R^2 scoring
    r2_scores = cross_val_score(final_model, X_sub, y_sub, cv=cv, scoring='r2')
    r2_mean = np.mean(r2_scores)
    r2_std = np.std(r2_scores)

    # MSE scoring (neg_mean_squared_error returns negative MSE => we multiply by -1)
    neg_mse_scores = cross_val_score(final_model, X_sub, y_sub, cv=cv, scoring='neg_mean_squared_error')
    mse_scores = -neg_mse_scores
    mse_mean = np.mean(mse_scores)
    mse_std = np.std(mse_scores)

    # 5) Optionally, fit the final model on the entire subset once
    final_model.fit(X_sub, y_sub)

    # 6) Create a dict of performance metrics
    performance_dict = {
        "r2_mean": r2_mean,
        "r2_std": r2_std,
        "mse_mean": mse_mean,
        "mse_std": mse_std
    }

    return final_model, selected_features, performance_dict

def load_feature_table(session_folder):
    """
    Load the feature table from the specified session folder.

    Parameters:
        session_folder (str): Path to the session folder.

    Returns:
        feature_table (pd.DataFrame): Feature table for the session.
    """
    feature_table_path = os.path.join(session_folder, "feature_table.csv")
    feature_table = pd.read_csv(feature_table_path)

    return feature_table