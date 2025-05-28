import glob
import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import seaborn as sns
from PyEMD import EMD
from fooof import FOOOF
from matplotlib.gridspec import GridSpec
from padasip.filters import FilterRLS
from scipy import signal
from scipy.stats import pearsonr

from src.core.ExpProcessor import ExpProcessor
from src.data_management.DataObj import DataObj


def extract_blink_segment(baseline_tuple, fs=250):
    """
    Extract the blink segment from the calibration data

    Parameters:
    - baseline_tuple: Tuple containing (data, time, triggers, trigger_times)
    - fs: Sampling frequency in Hz

    Returns:
    - blink_data: The extracted blink segment
    - blink_time: Time vector for the blink segment
    """
    baseline_data, baseline_time, triggers, trigger_times, _ = baseline_tuple

    # Find the index of trigger 17 (start of face movement)
    start_idx = np.where(triggers == 17)[0]
    if len(start_idx) == 0:
        print("Trigger 17 not found in baseline data")
        return None, None

    # Find the index of trigger 14 (end of blink segment) after trigger 17
    end_idx = np.where(triggers == 14)[0]
    if len(end_idx) == 0:
        print("Trigger 14 not found in baseline data")
        return None, None

    # Find the first trigger 14 that comes after trigger 17
    valid_end_indices = end_idx[end_idx > start_idx[0]]
    if len(valid_end_indices) == 0:
        print("No trigger 14 found after trigger 17")
        return None, None

    # Get the times corresponding to the start and end triggers
    start_time = trigger_times[start_idx[0]]
    end_time = trigger_times[valid_end_indices[0]]

    # Find the corresponding indices in the time vector
    start_sample = np.argmin(np.abs(baseline_time - start_time))
    end_sample = np.argmin(np.abs(baseline_time - end_time))

    # Extract the blink segment
    blink_data = baseline_data[start_sample:end_sample, :]
    blink_time = baseline_time[start_sample:end_sample] - baseline_time[start_sample]

    return blink_data, blink_time

def compare_trials_beta(processor, beta_trial_high, beta_trial_low, id_num, save_dir, fs = 250):
    """
    Compare EEG data across three conditions:
    1. Baseline (eyes-closed calibration)
    2. High cognitive load maze task
    3. Low cognitive load maze task

    This improved version focuses on better artifact handling and cognitive load metrics.

    Parameters:
    - processor: Data processor object
    - beta_trial_high: ID of the high cognitive load maze trial
    - beta_trial_low: ID of the low cognitive load maze trial

    Returns:
    - Visualizations comparing the three conditions
    """
    # ------------------------------------------------------------------------------
    # SECTION 1: DATA EXTRACTION
    # ------------------------------------------------------------------------------
    # Extract trials data
    print("Extracting trial data...")
    baseline_tuple = processor.extract_trials(status=2)  # Calibration (eyes-closed)
    beta_high_tuple = processor.extract_trials(status=1, period=beta_trial_high)  # High cognitive load
    beta_low_tuple = processor.extract_trials(status=1, period=beta_trial_low)

    if beta_trial_high == 1:
        blink_data, blink_time = extract_blink_segment(baseline_tuple)

    # Extract the eyes-closed segment from baseline using triggers
    baseline_time = baseline_tuple[1]  # time vector

    # Find indices where triggers 22 and 23 occur (eyes closed start/end)
    index_time = np.where(baseline_tuple[2] == [22, 23])

    # Extract the timestamps of these triggers
    time_start = baseline_tuple[3][index_time[0][0]]  # Start time
    time_end = baseline_tuple[3][index_time[0][1]]  # End time

    # Find corresponding indices in the time vector
    index_time_start = np.argmin(np.abs(baseline_time - time_start))
    index_time_end = np.argmin(np.abs(baseline_time - time_end))

    # Extract only the eyes-closed segment
    baseline_data = baseline_tuple[0][index_time_start:index_time_end, :]
    baseline_time = baseline_time[index_time_start:index_time_end]

    # Extract data for high and low cognitive load trials
    beta_high_data = beta_high_tuple[0]
    beta_high_time = beta_high_tuple[1]

    beta_low_data = beta_low_tuple[0]
    beta_low_time = beta_low_tuple[1]

    # Extract only EEG channels 13-16 (0-indexed: 12-15)
    # These are typically the frontal/central channels most relevant for cognitive load
    beta_channels = slice(12, 16)

    baseline_channels_data = baseline_data[:, beta_channels]
    beta_high_channels_data = beta_high_data[:, beta_channels]
    beta_low_channels_data = beta_low_data[:, beta_channels]
    cheek_channel_data = beta_high_data[:, 10]

    # ------------------------------------------------------------------------------
    # SECTION 2: EMG ARTIFACT FILTERING USING EMDRLS
    # ------------------------------------------------------------------------------
    print("Applying EMDRLS filtering to remove EMG artifacts...")

    def calculate_performance_metrics(original_signal, filtered_signal, emg_free_mask, fs=250):
        """
        Calculate performance metrics as defined in the paper:
        1. SNR (Signal-to-Noise Ratio)
        2. RMSE (Root Mean Square Error) of power spectra
        3. MPSD (Mean Power Spectral Density) for brain rhythms

        Parameters:
        - original_signal: Original EEG signal with EMG artifacts
        - filtered_signal: Filtered EEG signal
        - emg_free_mask: Boolean array indicating EMG-free regions
        - fs: Sampling frequency in Hz

        Returns:
        - metrics: Dictionary containing performance metrics
        """
        metrics = {}

        # Define regions as in Figure 6 of the paper
        # A: Region with EMG contamination in original signal
        # B: EMG-free region in original signal
        # C: Filtered region corresponding to contaminated region A
        # D: Filtered region corresponding to EMG-free region B

        emg_present_mask = ~emg_free_mask

        # Find longest continuous segments of EMG and EMG-free regions
        def find_longest_segment(mask):
            segments = []
            current_segment = []
            for i, val in enumerate(mask):
                if val:
                    current_segment.append(i)
                elif current_segment:
                    segments.append(current_segment)
                    current_segment = []
            if current_segment:
                segments.append(current_segment)
            if not segments:
                return np.array([])
            return np.array(max(segments, key=len))

        # Find longest contaminated and EMG-free segments
        emg_segment = find_longest_segment(emg_present_mask)
        emg_free_segment = find_longest_segment(emg_free_mask)

        # Ensure we have found segments
        if len(emg_segment) == 0 or len(emg_free_segment) == 0:
            print("Warning: Could not find both EMG and EMG-free segments")
            # Use default segments if none found
            if len(emg_segment) == 0 and len(emg_free_segment) > 0:
                # Use first half of signal as EMG segment if none found
                emg_segment = np.arange(len(original_signal) // 2)
            elif len(emg_free_segment) == 0 and len(emg_segment) > 0:
                # Use second half of signal as EMG-free segment if none found
                emg_free_segment = np.arange(len(original_signal) // 2, len(original_signal))
            else:
                # If neither found, split signal in half
                emg_segment = np.arange(len(original_signal) // 2)
                emg_free_segment = np.arange(len(original_signal) // 2, len(original_signal))

        # Equalize segment lengths by taking the middle portion of the longer segment
        min_length = min(len(emg_segment), len(emg_free_segment))
        if len(emg_segment) > min_length:
            start = (len(emg_segment) - min_length) // 2
            emg_segment = emg_segment[start:start + min_length]
        if len(emg_free_segment) > min_length:
            start = (len(emg_free_segment) - min_length) // 2
            emg_free_segment = emg_free_segment[start:start + min_length]

        # Extract segments from signals
        region_A = original_signal[emg_segment]
        region_B = original_signal[emg_free_segment]
        region_C = filtered_signal[emg_segment]
        region_D = filtered_signal[emg_free_segment]

        # 1. SNR calculations (as per Equations 4-6 in the paper)
        # Calculate powers using Equation 4
        p_a = np.sum(region_A ** 2) / len(region_A)
        p_b = np.sum(region_B ** 2) / len(region_B)
        p_c = np.sum(region_C ** 2) / len(region_C)
        p_d = np.sum(region_D ** 2) / len(region_D)

        # SNR of input signal (Equation 5)
        snr_in = 10 * np.log10(p_b / p_a)

        # SNR of filtered signal (Equation 6)
        snr_out = 10 * np.log10(p_d / p_c)

        metrics['snr_in'] = snr_in
        metrics['snr_out'] = snr_out
        metrics['p_a'] = p_a
        metrics['p_b'] = p_b
        metrics['p_c'] = p_c
        metrics['p_d'] = p_d

        # 2. RMSE of power spectral densities (Equations 7-8 in the paper)
        # Using Burg's method (equivalent to AR method) as mentioned in the paper

        # Calculate PSDs
        f_emg_free, psd_emg_free = signal.welch(region_B, fs=fs, nperseg=min(256, len(region_B)))
        f_original, psd_original = signal.welch(region_A, fs=fs, nperseg=min(256, len(region_A)))
        f_filtered, psd_filtered = signal.welch(region_C, fs=fs, nperseg=min(256, len(region_C)))

        # Calculate RMSE as per Equations 7 and 8
        rmse_in = np.sqrt(np.mean((psd_emg_free - psd_original) ** 2))
        rmse_out = np.sqrt(np.mean((psd_emg_free - psd_filtered) ** 2))

        metrics['rmse_in'] = rmse_in
        metrics['rmse_out'] = rmse_out

        # 3. MPSD for brain rhythms (as described in the paper)
        # Define frequency bands for different brain rhythms (in Hz)
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, min(100, fs / 2 - 1))  # Upper limit depends on sampling frequency
        }

        # Calculate MPSD for each band
        rhythm_bands = {
            'emg_free': {},
            'original': {},
            'filtered': {}
        }

        # Get frequencies for the PSDs we calculated earlier
        freqs = f_emg_free

        for band_name, (low_freq, high_freq) in bands.items():
            # Find frequency indices for this band
            idx_band = np.where((freqs >= low_freq) & (freqs <= high_freq))[0]

            if len(idx_band) > 0:
                # Calculate mean power in this band (in μV as per the paper)
                rhythm_bands['emg_free'][band_name] = np.sqrt(np.mean(psd_emg_free[idx_band]))
                rhythm_bands['original'][band_name] = np.sqrt(np.mean(psd_original[idx_band]))
                rhythm_bands['filtered'][band_name] = np.sqrt(np.mean(psd_filtered[idx_band]))

        metrics['mpsd'] = rhythm_bands

        # Store segments for visualization if needed
        metrics['segments'] = {
            'emg_segment': emg_segment,
            'emg_free_segment': emg_free_segment,
            'region_A': region_A,
            'region_B': region_B,
            'region_C': region_C,
            'region_D': region_D
        }

        return metrics

    def emdrls_filter(eeg_signal, fs=250, return_metrics=True):
        """
           Implementation of EMDRLS (Empirical Mode Decomposition with Recursive Least Squares)
           for removing facial EMG artifacts from EEG signals, following the paper's methodology.

           Parameters:
           - eeg_signal: 1D numpy array containing the EEG signal with EMG artifacts
           - fs: Sampling frequency in Hz
           - return_metrics: Whether to return performance metrics

           Returns:
           - filtered_signal: Cleaned EEG signal
           - metrics: (Optional) Dictionary containing performance metrics if return_metrics=True
                      (SNR, RMSE, MPSD) as used in the original paper
           """

        # Step 1: Detect EMG-free regions using a moving average filter
        window_size = int(fs / 8)  # Window size equal to half sampling frequency
        envelope = np.zeros_like(eeg_signal)
        for i in range(len(eeg_signal)):
            start_idx = max(0, i - window_size + 1)
            envelope[i] = np.mean(np.abs(eeg_signal[start_idx:i + 1]))


        # Find EMG-free regions (segments below mean threshold)
        threshold = 1.5 * np.mean(envelope)
        emg_free_mask = envelope < threshold

        # Step 2: Decompose signal using EMD
        # Initialize EMD
        emd = EMD()

        # Perform EMD decomposition
        imfs = emd.emd(eeg_signal)

        # Apply soft thresholding to individual IMFs
        thresholded_imfs = []

        # Process each IMF individually
        for m, imf in enumerate(imfs):
            # Extract the EMG-free regions of this specific IMF
            if np.any(emg_free_mask):
                # Get this IMF's values in the EMG-free regions
                imf_noise_region = imf[emg_free_mask]
                # Calculate standard deviation of this IMF in the EMG-free regions
                threshold = np.std(imf_noise_region)
            else:
                # If no EMG-free regions found, use a conservative estimate
                threshold = np.std(imf)

            # Apply soft thresholding as per Equation 3 in the paper:
            # tIMFm·sign(IMFm) = (|IMFm| - tm)+
            imf_thresh = np.zeros_like(imf)
            mask = np.abs(imf) > threshold
            imf_thresh[mask] = np.sign(imf[mask]) * (np.abs(imf[mask]) - threshold)

            thresholded_imfs.append(imf_thresh)

        # Reconstruct the reference EMG noise signal
        emg_reference = np.sum(thresholded_imfs, axis=0)

        # Step 3: Apply RLS adaptive filtering using padasip
        # Set RLS filter parameters
        n = 8  # Filter order
        mu = 0.99  # Forgetting factor

        # Initialize RLS filter
        filt = FilterRLS(n=n, mu=mu, w="zeros")

        # Apply the filter
        filtered_signal = np.zeros_like(eeg_signal)

        for i in range(len(eeg_signal)):
            # For each sample, get the reference input (past n samples of EMG reference)
            if i < n:
                # Handle initial samples where we don't have enough history
                x = np.zeros(n)
                if i > 0:
                    x[-i:] = emg_reference[:i]
            else:
                x = emg_reference[i - n:i][::-1]  # Get last n samples in reverse order

            # Predict the EMG artifact component
            y = filt.predict(x)

            # Subtract the predicted EMG from the original signal
            filtered_signal[i] = eeg_signal[i] - y

            # Update the filter weights
            filt.adapt(eeg_signal[i], x)

        # If metrics are not requested, return just the filtered signal
        if not return_metrics:
            return filtered_signal

        # ---------------------------------------------------------------------------
        # Calculate performance metrics as in the original paper
        # ---------------------------------------------------------------------------

        channel_metrics = calculate_performance_metrics(
            eeg_signal,
            filtered_signal,
            emg_free_mask,
            fs
        )

        return filtered_signal, channel_metrics

    def apply_emdrls_to_channels(eeg_data, fs=250):
        """
        Apply EMDRLS filtering to multi-channel EEG data

        Parameters:
        - eeg_data: 2D numpy array (samples x channels)
        - fs: Sampling frequency in Hz

        Returns:
        - filtered_data: Cleaned EEG data
        """
        filtered_data = np.zeros_like(eeg_data)
        all_metrics = {}

        for ch in range(eeg_data.shape[1]):
            filtered_data[:, ch], channel_metrics = emdrls_filter(eeg_data[:, ch], fs)

            # Store metrics
            all_metrics[f'channel_{ch}'] = channel_metrics

        return filtered_data, all_metrics

    def plot_emdrls_comparison(beta_high_data, high_load_filtered, beta_channels, blink_data=None, blink_filtered=None,
                               fs=250, save_dir=None, beta_trial_high=1):
        """
        Create and save plots comparing EEG signals before and after EMDRLS filtering.

        Parameters:
        - beta_high_data: Original EEG data from high cognitive load trial (samples x channels)
        - high_load_filtered: Filtered EEG data from high cognitive load trial (samples x channels)
        - beta_channels: Slice object indicating which channels were selected (e.g., slice(12, 16))
        - blink_data: Original EEG data from blink segment (samples x channels), if beta_trial_high=1
        - blink_filtered: Filtered EEG data from blink segment (samples x channels), if beta_trial_high=1
        - fs: Sampling frequency in Hz
        - save_dir: Directory to save the figures
        - beta_trial_high: Trial number (if 1, also plot blink data)

        Returns:
        - None (saves figures to disk)
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        # Create time vectors for the signals
        high_load_time = np.arange(beta_high_data.shape[0]) / fs

        # Number of channels to plot
        n_channels = beta_high_data.shape[1]

        # Get channel names (assuming the beta_channels slice starts from the actual channel indices)
        channel_names = [f"Channel {i + 1 + beta_channels.start}" for i in range(n_channels)]

        # Set up colors for the plots
        original_colors = ['#2E86C1', '#7D3C98', '#D35400', '#17A589']
        filtered_colors = ['#85C1E9', '#D2B4DE', '#F5B041', '#73C6B6']

        # Function to trim large artifacts at the beginning/end that affect scaling
        def trim_extreme_artifacts(data, time, threshold_std=3, padding=0.1):
            """
            Trim beginning/end artifacts that significantly affect scaling

            Parameters:
            - data: The signal data (samples x channels)
            - time: Time vector
            - threshold_std: Number of standard deviations to consider as extreme
            - padding: Padding in seconds to skip after initial artifacts

            Returns:
            - trimmed_data: Data with extreme artifacts removed
            - trimmed_time: Corresponding time vector
            - trim_indices: Indices of the trimmed data in the original array
            """
            n_samples, n_channels = data.shape

            # Find the initial index after extreme artifacts at the beginning
            start_idx = 0
            padding_samples = int(padding * fs)

            for ch in range(n_channels):
                # Calculate the signal mean and std excluding the potential artifact regions
                middle_region = data[padding_samples:n_samples - padding_samples, ch]
                signal_mean = np.mean(middle_region)
                signal_std = np.std(middle_region)

                # Define the threshold
                threshold = signal_mean + threshold_std * signal_std
                neg_threshold = signal_mean - threshold_std * signal_std

                # Find where the signal at the beginning is within thresholds
                for i in range(padding_samples):
                    if (data[i, ch] > threshold or data[i, ch] < neg_threshold):
                        if i + padding_samples > start_idx:
                            start_idx = i + padding_samples

            # Find the final index before extreme artifacts at the end
            end_idx = n_samples
            for ch in range(n_channels):
                # Define the threshold
                middle_region = data[padding_samples:n_samples - padding_samples, ch]
                signal_mean = np.mean(middle_region)
                signal_std = np.std(middle_region)
                threshold = signal_mean + threshold_std * signal_std
                neg_threshold = signal_mean - threshold_std * signal_std

                # Find where the signal at the end is within thresholds
                for i in range(n_samples - 1, n_samples - padding_samples - 1, -1):
                    if (data[i, ch] > threshold or data[i, ch] < neg_threshold):
                        if i - padding_samples < end_idx:
                            end_idx = i - padding_samples

            # Ensure we have some reasonable amount of data
            if end_idx - start_idx < fs:  # At least 1 second of data
                start_idx = 0
                end_idx = n_samples
                print("Warning: Couldn't trim artifacts without losing too much data.")

            # Return the trimmed data and time
            trim_indices = slice(start_idx, end_idx)
            return data[trim_indices, :], time[trim_indices], trim_indices

        # Trim artifacts that affect scaling
        trimmed_high_data, trimmed_high_time, high_indices = trim_extreme_artifacts(beta_high_data, high_load_time)
        trimmed_high_filtered = high_load_filtered[high_indices, :]

        if blink_data is not None and blink_filtered is not None:
            blink_time = np.arange(blink_data.shape[0]) / fs
            trimmed_blink_data, trimmed_blink_time, blink_indices = trim_extreme_artifacts(blink_data, blink_time)
            trimmed_blink_filtered = blink_filtered[blink_indices, :]

        # Create figure for high cognitive load comparison
        plt.figure(figsize=(15, 10))

        # Title for the figure
        plt.suptitle('EMDRLS Filtering Comparison - High Cognitive Load Trial', fontsize=16, fontweight='bold')

        # Create subplots for each channel
        for ch in range(n_channels):
            plt.subplot(n_channels, 1, ch + 1)

            # Plot original and filtered signals
            plt.plot(trimmed_high_time, trimmed_high_data[:, ch], color=original_colors[ch],
                     label='Original Signal', linewidth=1.5)
            plt.plot(trimmed_high_time, trimmed_high_filtered[:, ch], color=filtered_colors[ch],
                     label='EMDRLS Filtered', linewidth=1.5)

            # Add labels and legend
            plt.title(channel_names[ch], fontweight='bold')
            plt.ylabel('Amplitude (μV)')

            # Only add x-label for the bottom subplot
            if ch == n_channels - 1:
                plt.xlabel('Time (s)')

            # Add grid
            plt.grid(True, alpha=0.3, linestyle='--')

            # Add legend for first channel only
            if ch == 0:
                plt.legend(loc='upper right')

        # Adjust spacing
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

        # Save the figure
        if save_dir:
            high_load_fig_path = os.path.join(save_dir, f'emdrls_high_load_comparison.png')
            plt.savefig(high_load_fig_path, dpi=300, bbox_inches='tight')
            print(f"Saved high cognitive load comparison to: {high_load_fig_path}")

        # If beta_trial_high is 1, also plot blink data comparison
        if beta_trial_high == 1 and blink_data is not None and blink_filtered is not None:
            # Create figure for blink comparison
            plt.figure(figsize=(15, 10))

            # Title for the figure
            plt.suptitle('EMDRLS Filtering Comparison - Blink Segment from Calibration', fontsize=16, fontweight='bold')

            # Create subplots for each channel
            for ch in range(n_channels):
                plt.subplot(n_channels, 1, ch + 1)

                # Plot original and filtered signals
                plt.plot(trimmed_blink_time, trimmed_blink_data[:, ch], color=original_colors[ch],
                         label='Original Signal', linewidth=1.5)
                plt.plot(trimmed_blink_time, trimmed_blink_filtered[:, ch], color=filtered_colors[ch],
                         label='EMDRLS Filtered', linewidth=1.5)

                # Add labels and legend
                plt.title(channel_names[ch], fontweight='bold')
                plt.ylabel('Amplitude (μV)')

                # Only add x-label for the bottom subplot
                if ch == n_channels - 1:
                    plt.xlabel('Time (s)')

                # Add grid
                plt.grid(True, alpha=0.3, linestyle='--')

                # Add legend for first channel only
                if ch == 0:
                    plt.legend(loc='upper right')

            # Adjust spacing
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)

            # Save the figure
            if save_dir:
                blink_fig_path = os.path.join(save_dir, f'emdrls_blink_comparison.png')
                plt.savefig(blink_fig_path, dpi=300, bbox_inches='tight')
                print(f"Saved blink segment comparison to: {blink_fig_path}")

        # plt.show()
        plt.close()

    # Apply EMDRLS filtering to all three conditions
    baseline_filtered, baseline_metrics = apply_emdrls_to_channels(baseline_channels_data, fs)
    high_load_filtered, high_load_metrics = apply_emdrls_to_channels(beta_high_channels_data, fs)
    low_load_filtered, low_load_metrics = apply_emdrls_to_channels(beta_low_channels_data, fs)

    # Add subject and maze information to metrics
    for ch in high_load_metrics:
        high_load_metrics[ch]['subject_id'] = id_num
        high_load_metrics[ch]['maze'] = beta_trial_high

    if beta_trial_high == 1:
        # Store original blink data for comparison
        blink_original = blink_data[:,beta_channels].copy()

        # Apply EMDRLS filtering to blink data
        blink_filtered, blink_metrics  = apply_emdrls_to_channels(blink_data[:,beta_channels], fs)

        for ch in blink_metrics:
            blink_metrics[ch]['subject_id'] = id_num
            blink_metrics[ch]['maze'] = 0  # Indicate blinking

        # Plot and save the comparison figures
        plot_emdrls_comparison(
            beta_high_data=beta_high_channels_data,
            high_load_filtered=high_load_filtered,
            beta_channels=beta_channels,
            blink_data=blink_original,
            blink_filtered=blink_filtered,
            fs=fs,
            save_dir=save_dir,
            beta_trial_high=beta_trial_high
        )

        EMDRLS_matrics = [high_load_metrics, blink_metrics]

    else:
        # Plot and save just the high load comparison
        plot_emdrls_comparison(
            beta_high_data=beta_high_channels_data,
            high_load_filtered=high_load_filtered,
            beta_channels=beta_channels,
            fs=fs,
            save_dir=save_dir,
            beta_trial_high=beta_trial_high
        )

        EMDRLS_matrics = [high_load_metrics]



    # Store original data for comparison
    baseline_original = baseline_channels_data.copy()
    high_load_original = beta_high_channels_data.copy()
    low_load_original = beta_low_channels_data.copy()

    # ------------------------------------------------------------------------------
    # SECTION 3: FREQUENCY BAND ANALYSIS
    # ------------------------------------------------------------------------------
    print("Performing improved frequency band analysis...")

    # Define frequency bands with clear separation
    freq_bands = {
        # 'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 12.5),
        'beta': (12.5, 30),
        'gamma': (30, 45),
        'emg': (45, 100)  # Higher frequencies likely to contain EMG
    }

    print("Finding high cognitive load segment...")
    times, beta_power, emg_power, beta_emg_ratio = analyze_beta_emg_ratio(high_load_filtered, save_dir, fs)
    _, cheek_beta_pw, cheek_emg_pw, _ = analyze_beta_emg_ratio(cheek_channel_data.reshape(-1, 1), save_dir, fs, flag = False)

    cheek_beta_pw = cheek_beta_pw[0]  # flatten list-of-one
    cheek_emg_pw = cheek_emg_pw[0]

    corr_feats = emg_leakage_metrics(beta_power,
                                       emg_power,
                                       cheek_beta_pw,
                                       cheek_emg_pw)

    beta_feats = extract_beta_metrics(beta_power, beta_emg_ratio)

    rows = []

    for bf, cf in zip(beta_feats, corr_feats):
        # glue the two dicts together
        feat_row = {**bf, **cf,
                    "subject": id_num,
                    "maze": maze,
                    "tlx": data.trail_cog_nasa[maze]}
        rows.append(feat_row)

    for i in range(4):
        fig = plot_beta_emg_data(times,
                             beta_power,
                             emg_power,
                             beta_emg_ratio,
                             save_dir,
                             channel=i,
                             threshold=0)
    #     fig.show()

    segments = find_best_segments(times, beta_emg_ratio)

    empty_count_alt = sum(1 for segment in segments if not segment)
    non_empty_segments = [segment for segment in segments if segment]

    if empty_count_alt == len(segments):
        print("No segments found for any channel.")
        return rows, EMDRLS_matrics
    elif empty_count_alt > 0:
        print(f"Warning: {empty_count_alt} channels have no segments.")

    coherence_segment = max(non_empty_segments,
                            key=lambda segment: segment[0]['mean_ratio'] if segment else float('-inf'))



    figs = plot_best_segments_spectrograms(high_load_filtered, segments, save_dir)
    # for fig in figs:
    #     fig.show()


    high_load_segment, high_load_time, _, _ = extract_segment_data(high_load_filtered,
                                                                   coherence_segment[0],
                                                                   fs,
                                                                   window_size = 1.0,
                                                                   channel=3)

    high_load_freqs, high_load_psd = signal.welch(high_load_segment, fs=fs, nperseg=min(3*fs, len(high_load_segment)),
                              window='hann', scaling='density')

    # Find characteristic segments for each condition
    print("Finding baseline segment with clear alpha...")
    baseline_segment, baseline_time, baseline_powers, baseline_freqs, baseline_psd, \
        baseline_start_idx, baseline_end_idx, baseline_best_channel = \
        find_characteristic_segment(baseline_filtered, fs, segment_length=3.0, condition_type='baseline')

    print(
        f"Selected baseline segment: channel {baseline_best_channel + 13}, indices {baseline_start_idx}-{baseline_end_idx}")

    print("Finding low cognitive load segment...")
    low_load_segment, low_load_time, low_load_powers, low_load_freqs, low_load_psd, \
        low_load_start_idx, low_load_end_idx, low_load_best_channel = \
        find_characteristic_segment(low_load_filtered, fs, segment_length=3.0, condition_type='low_load')

    print(
        f"Selected low load segment: channel {low_load_best_channel + 13}, indices {low_load_start_idx}-{low_load_end_idx}")


    # Plot spectral comparison

    # Create comprehensive visualization of all three conditions
    plot_comprehensive_comparison(baseline_segment, baseline_time,
                                  low_load_segment, low_load_time,
                                  high_load_segment, high_load_time,
                                  baseline_freqs, baseline_psd,
                                  low_load_freqs, low_load_psd,
                                  high_load_freqs, high_load_psd,
                                  fs, freq_bands, save_dir)

    return rows, EMDRLS_matrics

def find_characteristic_segment(data, fs, segment_length=3.0, step=0.5, condition_type='baseline'):
        """
        Find a segment with the clearest cognitive markers for the given condition.
        Ensures that the desired frequency pattern is consistently present throughout the segment.

        Parameters:
        - data: EEG data (samples x channels)
        - fs: Sampling frequency
        - segment_length: Length of segment to find (in seconds)
        - step: Step size for moving window (in seconds)
        - condition_type: 'baseline', 'high_load', or 'low_load'

        Returns:
        - best_segment: The data segment with clearest markers
        - best_segment_time: Time vector for the segment
        - band_powers: Dictionary of band powers for the best segment
        - psd_freqs, psd_values: PSD data for plotting
        - start_idx, end_idx: Original indices in the data
        - best_channel: The channel that showed the clearest markers
        """
        # Define frequency bands with clear separation
        freq_bands = {
            'theta': (4, 8),
            'alpha': (8, 12.5),
            'beta': (12.5, 30),
            'gamma': (30, 45),
            'emg': (45, 100)  # Higher frequencies likely to contain EMG
        }
        # run_spectrogram_explorer(data, fs)
        segment_samples = int(segment_length * fs)
        step_samples = int(step * fs)
        n_segments = (data.shape[0] - segment_samples) // step_samples + 1
        n_channels = data.shape[1]

        best_score = -np.inf
        best_segment_idx = 0
        best_channel = 0
        best_powers = None
        best_freqs = None
        best_psd = None

        # For time-frequency analysis - use smaller windows to track changes
        nperseg = int(fs * 0.25)  # 250ms segments
        noverlap = int(nperseg * 0.75)  # 75% overlap
        n_time_bins = 6  # Approximately 6 time bins for a 3-second segment
        # creat ndarray name score_vec of zeroes n_cannels on n_segments
        score_vec = np.zeros((n_channels, n_segments))

        for i in range(n_segments):
            start_idx = i * step_samples
            end_idx = start_idx + segment_samples

            if end_idx > data.shape[0]:
                continue

            # Extract segment
            segment = data[start_idx:end_idx, :]

            # Analyze each channel separately
            for ch in range(n_channels):
                channel_data = segment[:, ch]

                # 1. Overall PSD for the segment
                freqs, psd = signal.welch(channel_data, fs=fs, nperseg=segment_samples,
                                          window='hann', scaling='density')

                # Calculate overall band powers
                band_powers = {}
                for band_name, (low_freq, high_freq) in freq_bands.items():
                    idx = np.logical_and(freqs >= low_freq, freqs <= high_freq)
                    if np.any(idx):
                        band_powers[band_name] = 10 * np.log10(np.mean(psd[idx]) + 1e-10)
                    else:
                        band_powers[band_name] = float('nan')

                # 2. Time-frequency analysis
                f, t, Sxx = signal.spectrogram(channel_data, fs, window='hann', nperseg=nperseg, noverlap=noverlap)
                Sxx = 10*np.log10(Sxx + 1e-10)  # Convert to dB

                # Define the pattern we're looking for based on condition
                if condition_type == 'baseline':
                    # For each time bin, check if alpha > other bands AND alpha > EMG
                    alpha_higher_count = 0  # Count how many time bins have alpha dominance

                    for t_idx in range(len(t)):
                        other_powers = []
                        for band in freq_bands:
                            if band == 'alpha':
                                alpha_power = np.mean(Sxx[np.logical_and(f >= freq_bands[band][0],
                                                                         f <= freq_bands[band][1]), t_idx])
                            else:
                                band_power = np.mean(Sxx[np.logical_and(f >= freq_bands[band][0],
                                                                             f <= freq_bands[band][1]), t_idx])
                                other_powers.append(band_power)

                        # Check if alpha is the dominant rhythm in this time bin AND higher than EMG
                        if alpha_power > max(other_powers):
                            alpha_higher_count += 1

                    # Convert to a proportion of time bins where the pattern holds
                    pattern_consistency = alpha_higher_count / len(t)

                elif condition_type == 'high_load':
                    # For each time bin, check if beta > alpha and beta > emg
                    beta_dominant_count = 0

                    for t_idx in range(len(t)):
                        # Get band powers for this time bin
                        # Calculate the sum of beta bands
                        beta_power = np.mean(Sxx[np.logical_and(f >= freq_bands['beta'][0],
                                                                     f <= freq_bands['beta'][1]), t_idx])

                        alpha_power = np.mean(Sxx[np.logical_and(f >= freq_bands['alpha'][0],
                                                                      f <= freq_bands['alpha'][1]), t_idx])

                        emg_power = np.mean(Sxx[np.logical_and(f >= freq_bands['emg'][0],
                                                                    f <= freq_bands['emg'][1]), t_idx])

                        # Check if beta > alpha and beta is not just an EMG leak
                        if beta_power > alpha_power and beta_power > emg_power * 0.5:
                            beta_dominant_count += 1

                    pattern_consistency = beta_dominant_count / len(t)

                    # Get beta and EMG power over time
                    beta_band = freq_bands['beta']
                    emg_band = freq_bands['emg']
                    beta_mask = np.logical_and(f >= beta_band[0], f <= beta_band[1])
                    emg_mask = np.logical_and(f >= emg_band[0], f <= emg_band[1])
                    beta_power = np.mean(Sxx[beta_mask, :], axis=0)
                    emg_power = np.mean(Sxx[emg_mask, :], axis=0)

                    # Calculate mean beta-to-EMG ratio
                    beta_emg_ratio = beta_power-emg_power  # Using dB difference
                    mean_ratio = np.mean(beta_emg_ratio)

                    # Calculate variability (lower is better)
                    ratio_variability = np.std(beta_emg_ratio)

                    score = mean_ratio - (2 * ratio_variability)
                    score_vec[ch, i] = score

                else:  # low_load
                    # For low load, we want moderate beta throughout
                    good_ratio_count = 0

                    for t_idx in range(len(t)):
                        beta_power = np.mean(Sxx[np.logical_and(f >= freq_bands['beta'][0],
                                                                         f <= freq_bands['beta'][1]), t_idx])
                        alpha_power = np.mean(Sxx[np.logical_and(f >= freq_bands['alpha'][0],
                                                                      f <= freq_bands['alpha'][1]), t_idx])

                        # We want a moderate beta/alpha ratio (not too high or low)
                        ratio = beta_power / (alpha_power + 1e-10)
                        if 0.8 < ratio < 2.0:  # Moderate ratio range
                            good_ratio_count += 1

                    pattern_consistency = good_ratio_count / len(t)

                # 3. Overall score combines total power and pattern consistency
                if condition_type == 'baseline':
                    # For baseline: High alpha relative to other bands
                    alpha = band_powers['alpha']
                    other_bands_sum = band_powers['theta'] + band_powers['beta']
                    emg = band_powers['emg']

                    # Alpha should be higher than other neural bands and EMG should be low
                    alpha_dominance = alpha - other_bands_sum / 4  # How much alpha stands out
                    emg_penalty = max(0, emg - alpha)  # Penalty for high EMG

                    # Final score combines overall dominance with pattern consistency
                    score = alpha_dominance - emg_penalty + (10 * pattern_consistency)

                elif condition_type == 'high_load':
                    # For high load: High beta relative to alpha with low EMG
                    beta = band_powers['beta']
                    alpha = band_powers['alpha']
                    emg = band_powers['emg']

                    # Beta should be higher than alpha, and EMG shouldn't be too high
                    beta_dominance = beta - alpha  # How much beta stands out vs alpha
                    emg_penalty = max(0, emg - beta)  # Penalty for EMG higher than beta

                    # Final score combines overall dominance with pattern consistency
                    n_score = beta_dominance - emg_penalty + (10 * pattern_consistency)

                else:  # low_load
                    # For low load: Moderate beta
                    beta = band_powers['beta']
                    alpha = band_powers['alpha']
                    emg = band_powers['emg']

                    # For low load, beta shouldn't be too high or too low compared to alpha
                    beta_balance = 5 - abs(beta - alpha - 1)  # Optimal is beta = alpha + 1
                    emg_penalty = max(0, emg - beta)  # Penalty for high EMG

                    # Final score combines overall balance with pattern consistency
                    score = beta_balance - emg_penalty + (10 * pattern_consistency)

                if score > best_score:
                    best_score = score
                    best_segment_idx = i
                    best_channel = ch
                    best_powers = band_powers
                    best_freqs = freqs
                    best_psd = psd

        # Get the best segment and start/end indices
        start_idx = best_segment_idx * step_samples
        end_idx = start_idx + segment_samples
        best_segment = data[start_idx:end_idx, best_channel].reshape(-1, 1)  # Keep 2D shape
        best_segment_time = np.arange(segment_samples) / fs

        return best_segment, best_segment_time, best_powers, best_freqs, best_psd, start_idx, end_idx, best_channel

def evaluate_aperiodic(fm, freqs):
    """
    Re‑evaluate FOOOF’s aperiodic model on an arbitrary frequency vector.

    Parameters
    ----------
    fm : fooof.FOOOF
        A fitted FOOOF object.
    freqs : array_like
        Frequencies (Hz) on which to evaluate the model.

    Returns
    -------
    ap_db : ndarray
        Aperiodic component in dB, same length as `freqs`.
    """
    ap = fm.get_params('aperiodic_params')
    freqs = np.asarray(freqs, dtype=float)

    if fm.aperiodic_mode == 'fixed':
        offset, exponent = ap
        lin_power = 10**offset / freqs**exponent

    elif fm.aperiodic_mode == 'knee':
        offset, knee, exponent = ap
        lin_power = 10**offset / (knee + freqs**exponent)

    else:      # should never happen
        raise ValueError(f"Unknown aperiodic mode: {fm.aperiodic_mode}")

    with np.errstate(invalid='ignore'):
        ap_db = 10 * np.log10(lin_power + 1e-30)    # convert to dB

    bad = ~np.isfinite(ap_db)
    if bad[0]:
        good_idx = np.flatnonzero(~bad)
        if len(good_idx) >= 2:
            g0, g1 = good_idx[:2]
            delta =  ap_db[g0] - ap_db[g1]
            decay_pow = 1
            last_amplitude = ap_db[g0]
            for i in range(g0 - 1, -1, -1):  # נעבור אחורה  g0-1, g0-2, ...
                ap_db[i] = last_amplitude + delta / (5 ** decay_pow)
                decay_pow += 1
                last_amplitude = ap_db[i]
                if not bad[i - 1]:
                    break

    return ap_db

def remove_aperiodic_fooof(Sxx, f, fit_range=(2, 40)):
    """
    Estimate and subtract the aperiodic 1/f component (in dB) from a
    power‑spectrum matrix.

    Parameters
    ----------
    Sxx : ndarray  (n_freqs, n_time)  power (linear units)
    f   : ndarray  (n_freqs,)        frequency vector (Hz)
    fit_range : tuple               frequency range (low, high) for the fit

    Returns
    -------
    aperiodic_db : ndarray (n_freqs,)   the fitted 1/f curve in dB
    """
    # Average across time – more stable FOOOF fit
    f_safe = np.where(f == 0, 1e-3, f)  # never pass f=0 to log10
    psd_avg_lin = Sxx.mean(axis=1)  # **linear** mean spectrum
    psd_avg_lin[psd_avg_lin <= 0] = 1e-30

    # indices of frequencies used for the fit
    idx = (f >= fit_range[0]) & (f <= fit_range[1])
    f_fit, p_fit = f[idx], psd_avg_lin[idx]

    # --- 1) try FOOOF ------------------------------------------------
    fm = FOOOF(max_n_peaks=8, aperiodic_mode='knee', verbose=False)
    try:
        fm.fit(f_fit, p_fit)
        aperiodic_db = evaluate_aperiodic(fm, f)  # `f` is the full spectrogram axis

        return aperiodic_db, fm
    except Exception as e:                                    # noqa: BLE001
        warnings.warn(f"FOOOF failed ({e})")

        aperiodic_db = np.zeros_like(f)
    return aperiodic_db, fm

def analyze_beta_emg_ratio(eeg_data, save_dir, fs=250, window_size=1.0, step_size=0.02, flag = True):
    """
    Analyze the ratio between beta and EMG activity over time using STFT

    Parameters:
    - eeg_data: EEG data (samples x channels)
    - fs: Sampling frequency in Hz (default: 250 Hz)
    - window_size: Window size for STFT in seconds (default: 1.0 second)
    - step_size: Step size for STFT in seconds (default: 0.02 seconds = 20 ms)

    Returns:
    - times: Array of time points
    - beta_power: List of arrays with beta power for each channel
    - emg_power: List of arrays with EMG power for each channel
    - beta_emg_ratio: List of arrays with beta/EMG ratio for each channel
    """
    # Define frequency bands
    freq_bands = {
        'beta': (12.5, 30),
        'emg': (45, 100)
    }

    aperiodic_range = (2, 40)

    # Convert window and step size to samples
    nperseg = int(window_size * fs)
    noverlap = nperseg - int(step_size * fs)

    # Number of channels
    n_channels = eeg_data.shape[1]

    # Initialize results
    beta_power = []
    emg_power = []
    beta_emg_ratio = []
    fm_models = []
    times = None

    # Process each channel
    for ch in range(n_channels):
        channel_data = eeg_data[:, ch]

        # Compute STFT
        f, t, Sxx = signal.spectrogram(
            channel_data,
            fs=fs,
            window='hann',
            nperseg=nperseg,
            noverlap=nperseg-1,
            scaling='density'
        )

        ap_db, fm_model = remove_aperiodic_fooof(Sxx, f, aperiodic_range)
        fm_models.append(fm_model)

        # Convert to dB
        Sxx_db = 10 * np.log10(Sxx + 1e-10) - ap_db[:, None]

        # Store time points for reference
        if times is None:
            times = t

        # Calculate average power in beta and EMG bands over time
        beta_mask = np.logical_and(f >= freq_bands['beta'][0], f <= freq_bands['beta'][1])
        emg_mask = np.logical_and(f >= freq_bands['emg'][0], f <= freq_bands['emg'][1])

        # Average across the frequency dimension for each time point
        ch_beta_power = np.mean(Sxx_db[beta_mask], axis=0)
        ch_emg_power = np.mean(Sxx_db[emg_mask], axis=0)

        # Calculate beta-EMG ratio (difference in dB = ratio)
        ch_beta_emg_ratio = ch_beta_power - ch_emg_power

        # Store results
        beta_power.append(ch_beta_power)
        emg_power.append(ch_emg_power)
        beta_emg_ratio.append(ch_beta_emg_ratio)

    if flag:
        # 2×2 figure with annotated FOOOF fits
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()

        for ch, (fm, ax) in enumerate(zip(fm_models, axes)):
            fm.plot(ax=ax, annotate_peaks=False, plt_log=False)
            ax.set_title(f"Channel {ch + 13}")

        fig.suptitle("FOOOF aperiodic fits (β channels)", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(save_dir, "fooof_fits.png"), dpi=300)
        plt.close(fig)

    return times, beta_power, emg_power, beta_emg_ratio

def plot_beta_emg_data(times, beta_power, emg_power, beta_emg_ratio, save_dir, channel=0, threshold=0):
    """
    Plot beta and EMG power over time for a specific channel

    Parameters:
    - times: Array of time points
    - beta_power: List of arrays with beta power for each channel
    - emg_power: List of arrays with EMG power for each channel
    - beta_emg_ratio: List of arrays with beta/EMG ratio for each channel
    - channel: Channel index to plot (default: 0)
    - threshold: Threshold for ratio highlighting (default: 0)

    Returns:
    - fig: The matplotlib figure object
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot beta and EMG power
    ax1.plot(times, beta_power[channel], 'b-', label='Beta Power')
    ax1.plot(times, emg_power[channel], 'r-', label='EMG Power')
    ax1.set_ylabel('Power (dB)')
    ax1.set_title(f'Channel {channel + 1}: Beta and EMG Power Over Time')
    ax1.legend()
    ax1.grid(True)

    # Plot beta-EMG ratio
    ax2.plot(times, beta_emg_ratio[channel], 'g-')
    ax2.axhline(y=threshold, color='k', linestyle='--', label=f'Threshold ({threshold} dB)')

    # Highlight regions where ratio > threshold
    mask = beta_emg_ratio[channel] > threshold

    # Find contiguous regions
    region_starts = []
    region_ends = []

    if len(mask) > 0:
        # Add start of first region if it begins with True
        if mask[0]:
            region_starts.append(0)

        # Find transitions
        for i in range(1, len(mask)):
            if mask[i] and not mask[i - 1]:  # False to True transition
                region_starts.append(i)
            elif not mask[i] and mask[i - 1]:  # True to False transition
                region_ends.append(i - 1)

        # Add end of last region if it ends with True
        if mask[-1]:
            region_ends.append(len(mask) - 1)

    # Highlight each region
    for start, end in zip(region_starts, region_ends):
        ax2.axvspan(times[start], times[end], alpha=0.3, color='green')

    # Calculate and display stats
    percent_above = np.mean(mask) * 100
    if np.any(mask):
        mean_ratio_above = np.mean(beta_emg_ratio[channel][mask])
    else:
        mean_ratio_above = 0

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Beta-EMG Ratio (dB)')
    ax2.set_title(f'Beta-EMG Ratio Over Time (Green: Above {threshold} dB, {percent_above:.1f}% of time)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/channel_{channel + 1}_beta_emg_analysis.png', dpi=300)
    plt.close(fig)
    return fig

def find_best_segments(times, beta_emg_ratio, min_duration=1.0, threshold=0):
    """
    Find continuous segments where beta-EMG ratio exceeds the threshold

    Parameters:
    - times: Array of time points
    - beta_emg_ratio: List of arrays with beta/EMG ratio for each channel
    - min_duration: Minimum segment duration in seconds (default: 1.0 second)
    - threshold: Threshold for ratio (default: 0 dB)

    Returns:
    - segments: List of dictionaries with segment information for each channel
    """
    n_channels = len(beta_emg_ratio)
    segments = []

    for ch in range(n_channels):
        channel_segments = []
        ratio = beta_emg_ratio[ch]
        mask = ratio > threshold

        # Time step between consecutive points
        if len(times) > 1:
            dt = times[1] - times[0]
        else:
            dt = 0

        # Minimum number of consecutive points to form a segment
        min_points = int(min_duration / dt) if dt > 0 else 1

        # Find contiguous regions
        if len(mask) > 0:
            # Initialize variables for segment tracking
            in_segment = False
            start_idx = 0

            for i in range(len(mask)):
                if mask[i] and not in_segment:
                    # Start of a new segment
                    start_idx = i
                    in_segment = True
                elif not mask[i] and in_segment:
                    # End of a segment
                    end_idx = i - 1

                    # Check if segment meets minimum duration
                    if end_idx - start_idx + 1 >= min_points:
                        segment = {
                            'start_time': times[start_idx],
                            'end_time': times[end_idx],
                            'duration': times[end_idx] - times[start_idx],
                            'mean_ratio': np.mean(ratio[start_idx:end_idx + 1]),
                            'max_ratio': np.max(ratio[start_idx:end_idx + 1]),
                            'start_idx': start_idx,
                            'end_idx': end_idx
                        }
                        channel_segments.append(segment)

                    in_segment = False

            # Handle case where the last segment extends to the end
            if in_segment:
                end_idx = len(mask) - 1

                # Check if segment meets minimum duration
                if end_idx - start_idx + 1 >= min_points:
                    segment = {
                        'start_time': times[start_idx],
                        'end_time': times[end_idx],
                        'duration': times[end_idx] - times[start_idx],
                        'mean_ratio': np.mean(ratio[start_idx:end_idx + 1]),
                        'max_ratio': np.max(ratio[start_idx:end_idx + 1]),
                        'start_idx': start_idx,
                        'end_idx': end_idx
                    }
                    channel_segments.append(segment)

        # Sort segments by mean ratio (highest first)
        channel_segments.sort(key=lambda x: x['mean_ratio'], reverse=True)
        segments.append(channel_segments)

    return segments

def extract_segment_data(eeg_data, segment, fs, window_size, channel=0):
    """
    Extract the raw EEG data corresponding to a segment

    Parameters:
    - eeg_data: Original EEG data array (samples x channels)
    - segment: Segment dictionary from find_best_segments
    - fs: Sampling frequency in Hz
    - window_size: Window size used in STFT (in seconds)
    - step_size: Step size used in STFT (in seconds)
    - channel: Channel index (default: 0)

    Returns:
    - segment_data: Raw EEG data for the segment
    - segment_times: Time vector for the segment (in seconds)
    """
    # Calculate STFT parameters in samples
    nperseg = int(window_size * fs)

    # Get start and end times
    start_time = segment['start_time']
    end_time = segment['end_time']

    # Convert to sample indices in original data
    start_sample = int(start_time * fs)
    end_sample = int(end_time * fs) + nperseg  # Add window size to include full window

    # Ensure indices are within bounds
    start_sample = max(0, start_sample)
    end_sample = min(eeg_data.shape[0], end_sample)

    # Extract data
    segment_data = eeg_data[start_sample:end_sample, channel]

    # Create time vector
    segment_times = np.arange(len(segment_data)) / fs

    return segment_data, segment_times, start_sample, end_sample

def emg_leakage_metrics(beta_pw_db, emg_pw_db,
                        cheek_beta_db, cheek_emg_db,
                        r_thresh=0.30, p_thresh=0.05):
    """
    Check whether frontal‑beta time‑series are still tracking EMG.

    Parameters
    ----------
    beta_pw_db  : list[np.ndarray]   # 4 forehead channels, shape (n_bins,)
    emg_pw_db   : list[np.ndarray]   # same shape, forehead EMG power
    cheek_beta_db : np.ndarray        # cheek channel β‑band power, shape (n_bins,)
    cheek_emg_db  : np.ndarray        # cheek channel 45–100Hz EMG power, shape (n_bins,)
    r_thresh    : float             # |r| above which we call it "strong"
    p_thresh    : float             # significance cut‑off

    Returns
    -------
    Return a QC dict for every forehead channel:
   – r & p for three correlations
   – qc_emg_corr_ok  (True= channel accepted)

    """
    metrics = []

    for ch in range(len(beta_pw_db)):
        beta = beta_pw_db[ch]
        emg  = emg_pw_db[ch]

        r_be, p_be   = pearsonr(beta, emg)
        r_bcb, p_bcb = pearsonr(beta, cheek_beta_db)
        r_bce, p_bce = pearsonr(beta, cheek_emg_db)

        qc_pass = all((abs(r) < r_thresh or p > p_thresh)
                      for r, p in [(r_be, p_be),
                                   (r_bcb, p_bcb),
                                   (r_bce, p_bce)])

        metrics.append({
            "r_beta_emg":      r_be,  "p_beta_emg":      p_be,
            "r_beta_cheekBeta":  r_bcb, "p_beta_cheekBeta":  p_bcb,
            "r_beta_cheekEMG":   r_bce, "p_beta_cheekEMG":   p_bce,
            "qc_emg_corr_ok":  qc_pass
        })

    return metrics

def extract_beta_metrics(beta_power_db, ratio_db, ch_offset = 13):
    """
    Collapse the 1‑s time‑series that come out of `analyze_beta_emg_ratio`
    into two workload–ready scalars for every foreground channel.

    Parameters
    ----------
    beta_power_db : ndarray, shape (n_channels, n_windows)
        Beta‑band power time‑series in dB (already 10·log10).
    emg_power_db  : ndarray, shape (n_channels, n_windows)
        EMG‑band power time‑series in dB (same shape, unused here but
        passed in for completeness / future extensions).
    ratio_db      : ndarray, shape (n_channels, n_windows)
        beta_power_db–emg_power_db (dB difference) for each window.
    ch_offset     : int
        First channel number (for forehead channels 13‑16 the default is 13).

    Returns
    -------
    List[dict]
        One dictionary per channel, ready to be merged into your
        `rows` list before writing the CSV.  Keys:
            ─ 'channel'             : physical channel number
            ─ 'beta_mean_db'        : mean beta power across windows (dB)
            ─ 'pct_beta_above_3'    : %of windows with β–EMG>+3dB
    """
    metrics = []

    n_channels = len(beta_power_db)
    for ch in range(n_channels):
        beta_mean_db      = float(beta_power_db[ch].mean())
        pct_beta_above_3  = float((ratio_db[ch] > 3).mean() * 100)

        metrics.append({
            "channel":            ch + ch_offset,   # e.g. 13‑16
            "beta_mean_db":       beta_mean_db,
            "pct_beta_above_3":   pct_beta_above_3
        })

    return metrics

def plot_best_segments_spectrograms(eeg_data,
                                        segments,
                                        save_dir,
                                        fs=250,
                                        window_size=1.0,
                                        max_segments=4):
        """
        Plot spectrograms for the best segments from each channel.
        Creates 4 separate figures (one per channel), each containing a 2x2 grid of spectrograms
        showing the 4 best segments for that channel.

        Parameters:
        - eeg_data: Original EEG data array (samples x channels)
        - segments: List of segments from find_best_segments() for each channel
        - fs: Sampling frequency in Hz
        - window_size: Window size used in STFT (in seconds)
        - step_size: Step size used in STFT (in seconds)
        - max_segments: Maximum number of segments to plot per channel (default: 4)

        Returns:
        - figures: A list of matplotlib figure objects, one for each channel
        """
        n_channels = len(segments)
        figures = []  # List to store separate figures for each channel

        # Define frequency bands for visual reference
        freq_bands = {
            'beta': (12.5, 30),
            'emg': (45, 100)
        }

        # Process each channel (up to 4 channels)
        for ch in range(min(n_channels, 4)):
            # Get segments for this channel (up to max_segments)
            channel_segments = segments[ch][:max_segments]

            if len(channel_segments) > 0:
                # Create a new figure for this channel
                fig = plt.figure(figsize=(12, 10))

                # Create a 2x2 grid for this channel
                gs = GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.3)

                # Process each segment within this channel
                for i, segment in enumerate(channel_segments):
                    if i >= max_segments:
                        break

                    # Determine position in the 2x2 grid
                    row, col = i // 2, i % 2

                    # Extract the raw data for this segment
                    segment_data, segment_times, start_sample, end_sample = extract_segment_data(
                        eeg_data,
                        segment,
                        fs=fs,
                        window_size=window_size,
                        channel=ch
                    )

                    # Compute STFT with high time resolution
                    f, t, Sxx = signal.spectrogram(
                        segment_data,
                        fs=fs,
                        window='hann',
                        nperseg=int(fs),  # 1 second windows
                        noverlap=int(fs) - 1,  # High overlap for smooth visualization
                        scaling='density'
                    )

                    # Convert to dB
                    Sxx_db = 10 * np.log10(Sxx + 1e-10)

                    # Create subplot within the channel's grid
                    ax = fig.add_subplot(gs[row, col])

                    # Plot spectrogram
                    im = ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis',
                                       vmin=np.percentile(Sxx_db, 5), vmax=np.percentile(Sxx_db, 95))

                    # Add frequency band markers
                    for band_name, (low, high) in freq_bands.items():
                        color = 'r' if band_name == 'beta' else 'gray'
                        ax.axhline(y=low, color=color, linestyle='--', alpha=0.7)
                        ax.axhline(y=high, color=color, linestyle='--', alpha=0.7)

                    # Set title and labels
                    ax.set_title(f"Segment {i + 1}: β/EMG={segment['mean_ratio']:.2f}dB")

                    # Add axis labels
                    if row == 1:  # Bottom row
                        ax.set_xlabel("Time (s)")
                    if col == 0:  # First column
                        ax.set_ylabel("Frequency (Hz)")

                    # Set y-axis limits to focus on relevant frequencies
                    ax.set_ylim(0, 100)

                    # Add colorbar for each segment
                    plt.colorbar(im, ax=ax, label='Power (dB)')

                # Add a super title for the channel
                fig.suptitle(f"Channel {ch + 13} Best Segments", fontsize=16, fontweight='bold', y=0.98)

                # Adjust layout
                fig.tight_layout()
                fig.subplots_adjust(top=0.92)

                # Add figure to list
                figures.append(fig)

                # Save the figure
                fig.savefig(f"{save_dir}/channel_{ch + 13}_best_segments.png", dpi=300)
                plt.close(fig)
        return figures

def plot_comprehensive_comparison(baseline_segment, baseline_time,
                                  low_load_segment, low_load_time,
                                  high_load_segment, high_load_time,
                                  baseline_freqs, baseline_psd,
                                  low_load_freqs, low_load_psd,
                                  high_load_freqs, high_load_psd,
                                  fs, freq_bands, save_dir):
        """Plot a comprehensive comparison of the three conditions with time domain,
        spectrogram, and frequency domain visualizations."""

        # Create figure
        fig = plt.figure(figsize=(18, 14))

        # Define frequency limit for plots
        freq_limit = 100  # Hz

        # Define time and frequency for spectrograms
        nperseg = int(fs * 0.25)  # 250ms segments for good time-frequency resolution
        noverlap = int(nperseg * 0.75)  # 75% overlap

        # Normalize power spectra to dB
        baseline_psd_db = 10 * np.log10(baseline_psd + 1e-10)
        low_load_psd_db = 10 * np.log10(low_load_psd + 1e-10)
        high_load_psd_db = 10 * np.log10(high_load_psd + 1e-10)

        # Define colors for frequency bands
        band_colors = {
            'delta': 'purple',
            'theta': 'blue',
            'alpha': 'green',
            'beta': 'red',
            'gamma': 'magenta',
            'emg': 'gray'
        }

        # Row 1: Baseline
        # Time domain
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(baseline_time, baseline_segment, 'k-')
        ax1.set_title('Baseline (Eyes Closed) - Time Domain')
        ax1.set_ylabel('Amplitude (μV)')
        ax1.set_xlabel('Time (s)')
        ax1.grid(True, alpha=0.3)

        # Spectrogram
        ax2 = plt.subplot(3, 3, 2)
        # f, t, Sxx = signal.spectrogram(baseline_segment.flatten(), fs, nperseg=nperseg, noverlap=noverlap)
        f, t, Sxx = signal.spectrogram(
            baseline_segment.flatten(),
            fs=fs,
            window='hann',
            nperseg=int(fs),  # 1 second windows
            noverlap=int(fs) - 1,  # High overlap for smooth visualization
            scaling='density'
        )
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        im = ax2.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
        ax2.set_title('Baseline - Spectrogram')
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylim([0, freq_limit])

        # Add frequency band markers
        for band_name, (low, high) in freq_bands.items():
            ax2.axhline(y=low, color=band_colors[band_name], linestyle='--', alpha=0.7,
                        label=f"{band_name.capitalize()}")
            ax2.axhline(y=high, color=band_colors[band_name], linestyle='--', alpha=0.7)

        fig.colorbar(im, ax=ax2, label='Power (dB)')

        # Power spectrum
        ax3 = plt.subplot(3, 3, 3)
        ax3.plot(baseline_freqs, baseline_psd_db, 'k-')
        ax3.set_title('Baseline - Power Spectrum')
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Power (dB)')
        ax3.set_xlim([0, freq_limit])
        ax3.grid(True, alpha=0.3)

        # Shade frequency bands
        for band_name, (low, high) in freq_bands.items():
            if high <= freq_limit:  # Only show bands within display limit
                ax3.axvspan(low, high, alpha=0.2, color=band_colors.get(band_name, 'gray'),
                            label=f"{band_name.capitalize()}")

        ax3.legend(loc='upper right', fontsize='small')

        # Row 2: Low Cognitive Load
        # Time domain
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(low_load_time, low_load_segment, 'k-')
        ax4.set_title('Low Cognitive Load - Time Domain')
        ax4.set_ylabel('Amplitude (μV)')
        ax4.set_xlabel('Time (s)')
        ax4.grid(True, alpha=0.3)

        # Spectrogram
        ax5 = plt.subplot(3, 3, 5)
        # f, t, Sxx = signal.spectrogram(low_load_segment.flatten(), fs, nperseg=nperseg, noverlap=noverlap)
        f, t, Sxx = signal.spectrogram(
            low_load_segment.flatten(),
            fs=fs,
            window='hann',
            nperseg=int(fs),  # 1 second windows
            noverlap=int(fs) - 1,  # High overlap for smooth visualization
            scaling='density'
        )
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        im = ax5.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
        ax5.set_title('Low Cognitive Load - Spectrogram')
        ax5.set_ylabel('Frequency (Hz)')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylim([0, freq_limit])

        # Add frequency band markers
        for band_name, (low, high) in freq_bands.items():
            ax5.axhline(y=low, color=band_colors[band_name], linestyle='--', alpha=0.7,
                        label=f"{band_name.capitalize()}")
            ax5.axhline(y=high, color=band_colors[band_name], linestyle='--', alpha=0.7)

        fig.colorbar(im, ax=ax5, label='Power (dB)')

        # Power spectrum
        ax6 = plt.subplot(3, 3, 6)
        ax6.plot(low_load_freqs, low_load_psd_db, 'k-')
        ax6.set_title('Low Cognitive Load - Power Spectrum')
        ax6.set_xlabel('Frequency (Hz)')
        ax6.set_ylabel('Power (dB)')
        ax6.set_xlim([0, freq_limit])
        ax6.grid(True, alpha=0.3)

        # Shade frequency bands
        for band_name, (low, high) in freq_bands.items():
            if high <= freq_limit:  # Only show bands within display limit
                ax6.axvspan(low, high, alpha=0.2, color=band_colors.get(band_name, 'gray'),
                            label=f"{band_name.capitalize()}")

        ax6.legend(loc='upper right', fontsize='small')

        # Row 3: High Cognitive Load
        # Time domain
        ax7 = plt.subplot(3, 3, 7)
        ax7.plot(high_load_time, high_load_segment, 'k-')
        ax7.set_title('High Cognitive Load - Time Domain')
        ax7.set_ylabel('Amplitude (μV)')
        ax7.set_xlabel('Time (s)')
        ax7.grid(True, alpha=0.3)

        # Spectrogram
        ax8 = plt.subplot(3, 3, 8)
        # f, t, Sxx = signal.spectrogram(high_load_segment.flatten(), fs, nperseg=nperseg, noverlap=noverlap)
        f, t, Sxx = signal.spectrogram(
            high_load_segment.flatten(),
            fs=fs,
            window='hann',
            nperseg=int(fs),  # 1 second windows
            noverlap=int(fs) - 1,  # High overlap for smooth visualization
            scaling='density'
        )
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        im = ax8.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
        ax8.set_title('High Cognitive Load - Spectrogram')
        ax8.set_ylabel('Frequency (Hz)')
        ax8.set_xlabel('Time (s)')
        ax8.set_ylim([0, freq_limit])

        # Add frequency band markers
        for band_name, (low, high) in freq_bands.items():
            ax8.axhline(y=low, color=band_colors[band_name], linestyle='--', alpha=0.7,
                        label=f"{band_name.capitalize()}")
            ax8.axhline(y=high, color=band_colors[band_name], linestyle='--', alpha=0.7)

        fig.colorbar(im, ax=ax8, label='Power (dB)')

        # Power spectrum
        ax9 = plt.subplot(3, 3, 9)
        ax9.plot(high_load_freqs, high_load_psd_db, 'k-')
        ax9.set_title('High Cognitive Load - Power Spectrum')
        ax9.set_xlabel('Frequency (Hz)')
        ax9.set_ylabel('Power (dB)')
        ax9.set_xlim([0, freq_limit])
        ax9.grid(True, alpha=0.3)

        # Shade frequency bands
        for band_name, (low, high) in freq_bands.items():
            if high <= freq_limit:  # Only show bands within display limit
                ax9.axvspan(low, high, alpha=0.2, color=band_colors.get(band_name, 'gray'),
                            label=f"{band_name.capitalize()}")

        ax9.legend(loc='upper right', fontsize='small')

        plt.tight_layout()
        # plt.savefig('comprehensive_comparison.png', dpi=300)
        # plt.show()
        fig.savefig(os.path.join(save_dir, 'comprehensive_comparison.png'), dpi=300)
        plt.close(fig)



# make a list
list_id = ['06', '09', '12', '13', '14', '16', '19', '20', '21']

# list_id = ['06']

beta_corr_path = "highload_features.csv"
EMDRLS_path = "EMDRLS_matrics.csv"

current_path = os.getcwd()
if 'src' in current_path:
    current_path = os.path.join(current_path, '..', '..')

# Get the base path (Cognitive_Load directory)
base_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))

# Define the main output folder for all EEG analysis plots
main_folder = os.path.join(base_path, "eeg_analysis_plots")

# Create the main folder if it doesn't exist yet
if not os.path.exists(main_folder):
    os.makedirs(main_folder)
    print(f"Created main output folder: {main_folder}")

all_rows = []
all_EMDRLS = []

if not os.path.exists(os.path.join(main_folder, beta_corr_path)):
    for id_num in list_id:
        print(f'Participant {id_num}')
        print('---')
        # Find all XDF files in the directory
        xdf_pattern = os.path.join(current_path, 'data', f'participant_{id_num}', 'S01', "*.xdf")
        xdf_files = glob.glob(xdf_pattern)
        xdf_files.sort()

        # Pass either a single file or the list of files to DataObj
        if len(xdf_files) == 1:
            data = DataObj(xdf_files[0])  # Pass single file path
            session_folder = os.path.dirname(xdf_files[0])
        else:
            data = DataObj(xdf_files)
            session_folder = os.path.dirname(xdf_files[0])

        # session_folder = os.path.dirname(xdf_name)
        if f'processor_{id_num}.pkl' not in os.listdir(session_folder):

            # Initialize ExpProcessor
            processor = ExpProcessor(
                emg_data=data.ElectrodeStream,
                trigger_stream=data.Trigger_Cog,
                fs=250,
                window_size=30.0,
                overlap=0.5,
                subject_id=data.subject_id,  # Pass the subject ID
                sorted_indices=data.sorted_indices,  # Pass the sorted indices,
                auto_process=False,
                path=session_folder
            )

            # Save the object to a file with pickle and subject
            with open(rf'{session_folder}\processor_{id_num}.pkl', 'wb') as f:
                pickle.dump(processor, f)

        else:
            # Load the object from the file
            with open(rf'{session_folder}\processor_{id_num}.pkl', 'rb') as f:
                processor = pickle.load(f)
            f.close()

        # Define the subject-specific folder
        subject_folder = os.path.join(main_folder, f"subject_{id_num}")

        # Create subject folder if it doesn't exist
        if not os.path.exists(subject_folder):
            os.makedirs(subject_folder)
            print(f"Created subject folder: {subject_folder}")

        # Save NASA-TLX for this subject
        fig = data.plot_nasa_tlx(show_figure=False)
        fig.savefig(os.path.join(subject_folder, "nasa_tlx.png"), dpi=300)
        plt.close(fig)

        # Set up low cognitive load trial ID (constant across mazes)
        beta_trial_low = -1
        # Iterate through each maze for this subject
        for maze in range(len(data.sorted_indices)):
            if maze == 0: # Skip the 9th maze same as beta_trial_low
                continue

            if beta_trial_low < 8:
                beta_trial_low += 1

            # Define the maze-specific folder
            maze_folder = os.path.join(subject_folder, f"maze_{maze}")

            # Create maze folder if it doesn't exist
            if not os.path.exists(maze_folder):
                os.makedirs(maze_folder)
                print(f"Created maze folder: {maze_folder}")

            # Run the EEG analysis for this subject and maze
            # Pass the maze_folder to use for saving plots
            print("***")
            print(f"Running EEG analysis for maze {maze}...")
            metrics, EMDRLS_matrics = compare_trials_beta(processor, maze, beta_trial_low, id_num, save_dir=maze_folder)
            all_rows.extend(metrics)
            if maze == 1:
                all_EMDRLS.append(EMDRLS_matrics[1])
            all_EMDRLS.append(EMDRLS_matrics[0])

    pd.DataFrame(all_rows).to_csv(os.path.join(main_folder, "highload_features.csv"), index=False)
    print("Saved", len(all_rows), "rows to highload_features.csv")
    with open(rf'{main_folder}\EMDRLS_matrics.pkl', 'wb') as f:
        pickle.dump(all_EMDRLS, f)
    print("Saved", len(all_EMDRLS), "all_EMDRLS to all_EMDRLS.pkl")
else:
    df = pd.read_csv(os.path.join(main_folder, beta_corr_path))
    with open(rf'{main_folder}\EMDRLS_matrics.pkl', 'rb') as b:
        all_EMDRLS = pickle.load(b)
    b.close()


def create_metric_tables(all_EMDRLS):
    """
    Organize EMDRLS metrics into separate tables for each statistical measure.

    Parameters:
    - all_EMDRLS: List of dictionaries, where each dictionary contains metrics for 4 channels
                 Each channel entry contains a dictionary with various metrics

    Returns:
    - Dictionary of DataFrames, one for each statistical measure
    """
    # Initialize dictionaries to store data for each basic metric
    metrics_tables = {
        'snr_in': [],
        'snr_out': [],
        'p_a': [],
        'p_b': [],
        'p_c': [],
        'p_d': [],
        'rmse_in': [],
        'rmse_out': []
    }

    # Initialize dictionaries for MPSD metrics (brain rhythms)
    mpsd_metrics = {}
    for rhythm in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
        for state in ['emg_free', 'original', 'filtered']:
            mpsd_metrics[f'mpsd_{rhythm}_{state}'] = []

    # Process each entry in all_EMDRLS
    for metrics_dict in all_EMDRLS:
        # Each metrics_dict contains data for one maze/condition
        # It has keys 'channel_0', 'channel_1', etc.

        for ch_key, ch_metrics in metrics_dict.items():
            # Extract channel number
            ch_num = int(ch_key.split('_')[1])

            # Get subject and maze info
            subject_id = ch_metrics.get('subject_id')
            maze = ch_metrics.get('maze')

            # Process basic metrics
            for metric_name in metrics_tables.keys():
                if metric_name in ch_metrics:
                    metrics_tables[metric_name].append({
                        'subject_id': subject_id,
                        'maze': maze,
                        'channel': ch_num,
                        'value': ch_metrics[metric_name]
                    })

            # Process MPSD metrics
            if 'mpsd' in ch_metrics:
                for rhythm in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
                    for state in ['emg_free', 'original', 'filtered']:
                        if rhythm in ch_metrics['mpsd'].get(state, {}):
                            mpsd_key = f'mpsd_{rhythm}_{state}'
                            mpsd_metrics[mpsd_key].append({
                                'subject_id': subject_id,
                                'maze': maze,
                                'channel': ch_num,
                                'value': ch_metrics['mpsd'][state][rhythm]
                            })

    # Convert lists to pandas DataFrames
    dfs = {}
    for metric_name, entries in metrics_tables.items():
        if entries:  # Only create DataFrame if we have data
            dfs[metric_name] = pd.DataFrame(entries)

    # Convert MPSD metrics to DataFrames
    for mpsd_key, entries in mpsd_metrics.items():
        if entries:  # Only create DataFrame if we have data
            dfs[mpsd_key] = pd.DataFrame(entries)

    return dfs

metric_tables = create_metric_tables(all_EMDRLS)

# Save each table to a CSV file
for metric_name, df in metric_tables.items():
    df.to_csv(os.path.join(main_folder, f"{metric_name}_table.csv"), index=False)


def create_emdrls_tables(all_EMDRLS, save_dir=None):
    """
    Create subject-specific tables similar to Table 1 from the paper,
    but only for EMDRLS results.

    Parameters:
    - all_EMDRLS: List of dictionaries containing EMDRLS metrics for each maze/condition
    - save_dir: Directory to save the tables (optional)

    Returns:
    - Dictionary of DataFrames, one per subject
    """

    # Group data by subject
    subject_data = defaultdict(lambda: defaultdict(list))

    for metrics_dict in all_EMDRLS:
        # Each metrics_dict contains data for one maze/condition with keys like 'channel_0', 'channel_1', etc.
        for ch_key, ch_metrics in metrics_dict.items():
            if 'subject_id' in ch_metrics:
                subject_id = ch_metrics['subject_id']
                ch_num = int(ch_key.split('_')[1])  # Extract channel number
                subject_data[subject_id][ch_num].append(ch_metrics)

    # Create tables for each subject
    subject_tables = {}

    for subject_id, channels_data in subject_data.items():
        table_data = []

        for ch_num in sorted(channels_data.keys()):
            ch_metrics_list = channels_data[ch_num]

            # Calculate statistics for each metric across mazes
            row_data = {
                'Channel': f'Ch {ch_num + 13}',  # Convert to physical channel numbers (13-16)
            }

            # Extract metrics we're interested in (similar to Table 1 in paper)
            metrics_to_extract = ['snr_in', 'snr_out', 'rmse_in', 'rmse_out', 'p_a', 'p_b', 'p_c', 'p_d']

            for metric_name in metrics_to_extract:
                values = [ch_metrics[metric_name] for ch_metrics in ch_metrics_list if metric_name in ch_metrics]

                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values, ddof=1)  # Sample standard deviation
                    row_data[f'{metric_name}_mean'] = mean_val
                    row_data[f'{metric_name}_std'] = std_val
                    row_data[f'{metric_name}_formatted'] = f"{mean_val:.3f} ± {std_val:.3f}"

            # Add MPSD metrics (brain rhythms)
            rhythm_metrics = ['mpsd_delta_emg_free', 'mpsd_theta_emg_free', 'mpsd_alpha_emg_free',
                              'mpsd_beta_emg_free', 'mpsd_gamma_emg_free',
                              'mpsd_delta_original', 'mpsd_theta_original', 'mpsd_alpha_original',
                              'mpsd_beta_original', 'mpsd_gamma_original',
                              'mpsd_delta_filtered', 'mpsd_theta_filtered', 'mpsd_alpha_filtered',
                              'mpsd_beta_filtered', 'mpsd_gamma_filtered']

            for metric_name in rhythm_metrics:
                values = []
                for ch_metrics in ch_metrics_list:
                    if 'mpsd' in ch_metrics:
                        # Parse the metric name to extract rhythm and state
                        parts = metric_name.split('_')
                        rhythm = parts[1]  # delta, theta, etc.
                        state = '_'.join(parts[2:])  # emg_free, original, filtered

                        if state in ch_metrics['mpsd'] and rhythm in ch_metrics['mpsd'][state]:
                            values.append(ch_metrics['mpsd'][state][rhythm])

                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values, ddof=1)
                    row_data[f'{metric_name}_mean'] = mean_val
                    row_data[f'{metric_name}_std'] = std_val
                    row_data[f'{metric_name}_formatted'] = f"{mean_val:.3f} ± {std_val:.3f}"

            table_data.append(row_data)

        # Create DataFrame
        df = pd.DataFrame(table_data)
        subject_tables[subject_id] = df

        # Save if directory provided
        if save_dir:
            import os
            df.to_csv(os.path.join(save_dir, f'subject_{subject_id}_emdrls_table.csv'), index=False)

    return subject_tables


def create_simplified_rmse_table(all_EMDRLS, save_dir=None):
    """
    Create a simplified table similar to Table 1 from the paper focusing on RMSE metrics.
    This matches the paper's format more closely.

    Returns tables with:
    - Channel column
    - Without filtering (RMSE_in)
    - EMDRLS (RMSE_out)
    """

    # First, reorganize data by subject
    subject_data = defaultdict(lambda: defaultdict(list))

    for metrics_dict in all_EMDRLS:
        for ch_key, ch_metrics in metrics_dict.items():
            if 'subject_id' in ch_metrics:
                subject_id = ch_metrics['subject_id']
                ch_num = int(ch_key.split('_')[1])
                subject_data[subject_id][ch_num].append(ch_metrics)

    subject_tables = {}

    for subject_id, channels_data in subject_data.items():
        table_rows = []

        for ch_num in sorted(channels_data.keys()):
            ch_metrics_list = channels_data[ch_num]

            # Extract RMSE values across all mazes for this channel
            rmse_in_values = [m['rmse_in'] for m in ch_metrics_list if 'rmse_in' in m]
            rmse_out_values = [m['rmse_out'] for m in ch_metrics_list if 'rmse_out' in m]

            if rmse_in_values and rmse_out_values:
                rmse_in_mean = np.mean(rmse_in_values)
                rmse_in_std = np.std(rmse_in_values, ddof=1)

                rmse_out_mean = np.mean(rmse_out_values)
                rmse_out_std = np.std(rmse_out_values, ddof=1)

                row = {
                    'Channel': f'Ch {ch_num + 13}',  # Convert to physical channel numbers
                    'Without_Filtering': f"{rmse_in_mean:.3f} ± {rmse_in_std:.3f}",
                    'EMDRLS': f"{rmse_out_mean:.3f} ± {rmse_out_std:.3f}"
                }
                table_rows.append(row)

        df = pd.DataFrame(table_rows)
        subject_tables[subject_id] = df

        if save_dir:
            import os
            df.to_csv(os.path.join(save_dir, f'subject_{subject_id}_rmse_table.csv'), index=False)

    return subject_tables


def create_brain_rhythms_table(all_EMDRLS, save_dir=None):
    """
    Create tables similar to Table 3/4 from the paper focusing on brain rhythms (MPSD).
    """

    subject_data = defaultdict(lambda: defaultdict(list))

    for metrics_dict in all_EMDRLS:
        for ch_key, ch_metrics in metrics_dict.items():
            if 'subject_id' in ch_metrics and 'mpsd' in ch_metrics:
                subject_id = ch_metrics['subject_id']
                ch_num = int(ch_key.split('_')[1])
                subject_data[subject_id][ch_num].append(ch_metrics)

    subject_tables = {}
    rhythms = ['delta', 'theta', 'alpha', 'beta', 'gamma']

    for subject_id, channels_data in subject_data.items():
        # Average across all channels for this subject
        rhythm_data = {rhythm: {'emg_free': [], 'filtered': []} for rhythm in rhythms}

        for ch_num, ch_metrics_list in channels_data.items():
            for ch_metrics in ch_metrics_list:
                if 'mpsd' in ch_metrics:
                    for rhythm in rhythms:
                        if rhythm in ch_metrics['mpsd'].get('emg_free', {}):
                            rhythm_data[rhythm]['emg_free'].append(ch_metrics['mpsd']['emg_free'][rhythm])
                        if rhythm in ch_metrics['mpsd'].get('filtered', {}):
                            rhythm_data[rhythm]['filtered'].append(ch_metrics['mpsd']['filtered'][rhythm])

        table_rows = []
        for rhythm in rhythms:
            emg_free_vals = rhythm_data[rhythm]['emg_free']
            filtered_vals = rhythm_data[rhythm]['filtered']

            if emg_free_vals and filtered_vals:
                emg_free_mean = np.mean(emg_free_vals)
                emg_free_std = np.std(emg_free_vals, ddof=1)

                filtered_mean = np.mean(filtered_vals)
                filtered_std = np.std(filtered_vals, ddof=1)

                row = {
                    'Rhythm': rhythm.capitalize(),
                    'EMG_Free_EEG': f"{emg_free_mean:.3f} ± {emg_free_std:.3f}",
                    'EMDRLS': f"{filtered_mean:.3f} ± {filtered_std:.3f}"
                }
                table_rows.append(row)

        df = pd.DataFrame(table_rows)
        subject_tables[subject_id] = df

        if save_dir:
            import os
            df.to_csv(os.path.join(save_dir, f'subject_{subject_id}_brain_rhythms_table.csv'), index=False)

    return subject_tables


def generate_all_tables(all_EMDRLS, save_dir):
    """
    Generate all the tables similar to the paper's format.
    """
    print("Generating RMSE tables (similar to Table 1)...")
    rmse_tables = create_simplified_rmse_table(all_EMDRLS, save_dir)

    print("Generating brain rhythms tables (similar to Tables 3-4)...")
    rhythm_tables = create_brain_rhythms_table(all_EMDRLS, save_dir)

    print("Generating comprehensive EMDRLS tables...")
    comprehensive_tables = create_emdrls_tables(all_EMDRLS, save_dir)

    return {
        'rmse_tables': rmse_tables,
        'rhythm_tables': rhythm_tables,
        'comprehensive_tables': comprehensive_tables
    }


tables = generate_all_tables(all_EMDRLS, main_folder)


def categorize_by_contamination_level(snr_in_value):
    """
    Categorize SNR_in values into contamination levels as per the paper
    """
    if snr_in_value >= -5:
        return 1
    elif snr_in_value >= -10:
        return 2
    elif snr_in_value >= -20:
        return 3
    elif snr_in_value >= -30:
        return 4
    else:
        return 5


def plot_contamination_vs_performance_by_channel(all_EMDRLS, subject_id, save_dir=None):
    """
    Create Figure 2-style plot but with channels instead of different filters

    For each subject:
    - X-axis: Contamination level (1-5)
    - Y-axis: SNR_out (performance after filtering)
    - Different lines: Different channels
    - Error bars: Standard deviation within each contamination level
    """

    # Extract data for this subject
    subject_data = defaultdict(lambda: defaultdict(list))  # [channel][contamination_level] = [snr_out_values]

    for metrics_dict in all_EMDRLS:
        for ch_key, ch_metrics in metrics_dict.items():
            if ch_metrics.get('subject_id') == subject_id:
                ch_num = int(ch_key.split('_')[1])

                if 'snr_in' in ch_metrics and 'snr_out' in ch_metrics:
                    contamination_level = categorize_by_contamination_level(ch_metrics['snr_in'])
                    subject_data[ch_num][contamination_level].append(ch_metrics['snr_out'])

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Different colors for each channel
    markers = ['o', 's', '^', 'D']

    for ch_idx, (ch_num, contamination_data) in enumerate(sorted(subject_data.items())):
        contamination_levels = []
        mean_snr_out = []
        std_snr_out = []

        for level in range(1, 6):  # Levels 1-5
            if level in contamination_data and len(contamination_data[level]) > 0:
                contamination_levels.append(level)
                mean_snr_out.append(np.mean(contamination_data[level]))
                std_snr_out.append(np.std(contamination_data[level]))

        if contamination_levels:  # Only plot if we have data
            # Plot line with error bars
            ax.errorbar(contamination_levels, mean_snr_out, yerr=std_snr_out,
                        marker=markers[ch_idx], color=colors[ch_idx],
                        label=f'Ch {ch_num + 13}', linewidth=2, markersize=8,
                        capsize=5, capthick=2)

    ax.set_xlabel('Contamination Level', fontsize=12)
    ax.set_ylabel('SNR out (dB)', fontsize=12)
    ax.set_title(f'Subject {subject_id}: EMDRLS Performance vs Contamination Level by Channel',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(['Level 1\n(≥ -5dB)', 'Level 2\n(-10 to -5dB)', 'Level 3\n(-20 to -10dB)',
                        'Level 4\n(-30 to -20dB)', 'Level 5\n(< -30dB)'], fontsize=10)

    plt.tight_layout()

    if save_dir:
        import os
        plt.savefig(os.path.join(save_dir, f'subject_{subject_id}_contamination_vs_performance.png'),
                    dpi=300, bbox_inches='tight')

    return fig


def plot_channel_performance_comparison(all_EMDRLS, subject_id, save_dir=None):
    """
    Create Figure 3-style plot showing before/after SNR by channel

    For each subject:
    - X-axis: Channel number
    - Y-axis: SNR (dB)
    - Two sets of bars/points: SNR_in (before) and SNR_out (after)
    - Can be box plots to show variation across mazes
    """

    # Extract data for this subject
    channel_data = defaultdict(lambda: {'snr_in': [], 'snr_out': []})

    for metrics_dict in all_EMDRLS:
        for ch_key, ch_metrics in metrics_dict.items():
            if ch_metrics.get('subject_id') == subject_id:
                ch_num = int(ch_key.split('_')[1])

                if 'snr_in' in ch_metrics and 'snr_out' in ch_metrics:
                    channel_data[ch_num]['snr_in'].append(ch_metrics['snr_in'])
                    channel_data[ch_num]['snr_out'].append(ch_metrics['snr_out'])

    # Prepare data for plotting
    channels = sorted(channel_data.keys())
    channel_labels = [f'Ch {ch + 13}' for ch in channels]

    # Create two subplots: bar plot and box plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Subplot 1: Bar plot with error bars
    snr_in_means = []
    snr_in_stds = []
    snr_out_means = []
    snr_out_stds = []

    for ch in channels:
        snr_in_means.append(np.mean(channel_data[ch]['snr_in']))
        snr_in_stds.append(np.std(channel_data[ch]['snr_in']))
        snr_out_means.append(np.mean(channel_data[ch]['snr_out']))
        snr_out_stds.append(np.std(channel_data[ch]['snr_out']))

    x = np.arange(len(channels))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, snr_in_means, width, yerr=snr_in_stds,
                    label='Before Filtering (SNR_in)', color='lightcoral',
                    capsize=5, alpha=0.8)
    bars2 = ax1.bar(x + width / 2, snr_out_means, width, yerr=snr_out_stds,
                    label='After EMDRLS (SNR_out)', color='lightblue',
                    capsize=5, alpha=0.8)

    ax1.set_xlabel('Channel', fontsize=12)
    ax1.set_ylabel('SNR (dB)', fontsize=12)
    ax1.set_title(f'Subject {subject_id}: SNR Comparison by Channel (Mean ± SD)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(channel_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax1.text(bar1.get_x() + bar1.get_width() / 2., height1 + snr_in_stds[i] + 0.5,
                 f'{height1:.1f}', ha='center', va='bottom', fontsize=9)
        ax1.text(bar2.get_x() + bar2.get_width() / 2., height2 + snr_out_stds[i] + 0.5,
                 f'{height2:.1f}', ha='center', va='bottom', fontsize=9)

    # Subplot 2: Box plot showing distribution
    box_data = []
    box_labels = []
    colors = []

    for ch in channels:
        box_data.extend([channel_data[ch]['snr_in'], channel_data[ch]['snr_out']])
        box_labels.extend([f'Ch {ch + 13}\nBefore', f'Ch {ch + 13}\nAfter'])
        colors.extend(['lightcoral', 'lightblue'])

    bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)

    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax2.set_xlabel('Channel', fontsize=12)
    ax2.set_ylabel('SNR (dB)', fontsize=12)
    ax2.set_title(f'Subject {subject_id}: SNR Distribution by Channel', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_dir:
        import os
        plt.savefig(os.path.join(save_dir, f'subject_{subject_id}_channel_comparison.png'),
                    dpi=300, bbox_inches='tight')

    return fig


def plot_improvement_summary(all_EMDRLS, subject_id, save_dir=None):
    """
    Create a summary plot showing the improvement achieved by EMDRLS
    """

    # Extract data for this subject
    channel_data = defaultdict(lambda: {'snr_in': [], 'snr_out': [], 'improvement': []})

    for metrics_dict in all_EMDRLS:
        for ch_key, ch_metrics in metrics_dict.items():
            if ch_metrics.get('subject_id') == subject_id:
                ch_num = int(ch_key.split('_')[1])

                if 'snr_in' in ch_metrics and 'snr_out' in ch_metrics:
                    snr_in = ch_metrics['snr_in']
                    snr_out = ch_metrics['snr_out']
                    improvement = snr_out - snr_in

                    channel_data[ch_num]['snr_in'].append(snr_in)
                    channel_data[ch_num]['snr_out'].append(snr_out)
                    channel_data[ch_num]['improvement'].append(improvement)

    # Create improvement plot
    fig, ax = plt.subplots(figsize=(10, 6))

    channels = sorted(channel_data.keys())
    channel_labels = [f'Ch {ch + 13}' for ch in channels]

    improvements = [np.mean(channel_data[ch]['improvement']) for ch in channels]
    improvement_stds = [np.std(channel_data[ch]['improvement']) for ch in channels]

    bars = ax.bar(channel_labels, improvements, yerr=improvement_stds,
                  color='green', alpha=0.7, capsize=5)

    ax.set_xlabel('Channel', fontsize=12)
    ax.set_ylabel('SNR Improvement (dB)', fontsize=12)
    ax.set_title(f'Subject {subject_id}: EMDRLS SNR Improvement by Channel',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    # Add value labels on bars
    for bar, improvement, std in zip(bars, improvements, improvement_stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + std + 0.2,
                f'{improvement:.1f} dB', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    if save_dir:
        import os
        plt.savefig(os.path.join(save_dir, f'subject_{subject_id}_improvement_summary.png'),
                    dpi=300, bbox_inches='tight')

    return fig


def create_all_subject_plots(all_EMDRLS, save_dir):
    """
    Create all plots for all subjects
    """
    # Get unique subject IDs
    subject_ids = set()
    for metrics_dict in all_EMDRLS:
        for ch_key, ch_metrics in metrics_dict.items():
            if 'subject_id' in ch_metrics:
                subject_ids.add(ch_metrics['subject_id'])

    print(f"Creating plots for {len(subject_ids)} subjects...")

    for subject_id in sorted(subject_ids):
        print(f"Processing Subject {subject_id}...")

        # Create contamination vs performance plot
        fig1 = plot_contamination_vs_performance_by_channel(all_EMDRLS, subject_id, save_dir)
        plt.show()

        # Create channel comparison plot
        fig2 = plot_channel_performance_comparison(all_EMDRLS, subject_id, save_dir)
        plt.show()

        # Create improvement summary plot
        fig3 = plot_improvement_summary(all_EMDRLS, subject_id, save_dir)
        plt.show()

        plt.close('all')  # Close figures to save memory

    print("All plots created successfully!")

create_all_subject_plots(all_EMDRLS, main_folder)



stat_indices = ['beta_mean_db', 'pct_beta_above_3', 'r_beta_emg', 'r_beta_cheekBeta',
                'r_beta_cheekEMG']

channels = df['channel'].unique()
subject = df['subject'].unique()

# Create dictionary to store results
results = {}
rows = []

for sub in subject:
    sub_results = {}

    for ch in channels:
        filtered_df = df[(df['subject'] == sub) & (df['channel'] == ch)]
        ch_results = {}

        for stat in stat_indices:
                r, p = pearsonr(filtered_df[stat], filtered_df['tlx'])
                ch_results[stat] = {'r': r, 'p': p}
                rows.append({
                    'subject': sub,
                    'channel': ch,
                    'statistic': stat,
                    'r': r,
                    'p': p,
                    'significant': p < 0.05
                })

        sub_results[ch] = ch_results

    results[sub] = sub_results

# Convert results to DataFrame for easier analysis
correlation_df = pd.DataFrame(rows)

# Save results to CSV
correlation_df.to_csv(os.path.join(main_folder, 'tlx_correlations.csv'), index=False)

# Create one figure per statistical index
for stat in stat_indices:
    # Filter data for this statistic
    stat_data = correlation_df[correlation_df['statistic'] == stat].copy()
    stat_data['r'] = abs(stat_data['r'])

    plt.figure(figsize=(10, 6))

    # Create a boxplot of correlation values by channel
    ax = sns.boxplot(x='channel', y='r', data=stat_data)

    # Add individual data points representing each subject's correlation
    sns.stripplot(x='channel', y='r', data=stat_data,
                  hue='significant', palette={True: 'red', False: 'black'},
                  size=6, alpha=0.7, jitter=True)

    # Set y-axis limits between 0 and 1 for better understanding
    plt.ylim(0, 1)

    # Add title and labels
    plt.title(f'Absolute Correlation between {stat} and TLX Score across Channels')
    plt.xlabel('Channel')
    plt.ylabel('Absolute Correlation Coefficient (|r|)')

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:  # Only add legend if there are items to show
        plt.legend(title='Significant (p<0.05)')

    # Add grid for easier reading
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Improve appearance
    plt.tight_layout()

    # Save each plot
    plt.savefig(f'{stat}_abs_correlation_by_channel.png', dpi=300)
    plt.show()

# Summary plot showing average correlations
plt.figure(figsize=(12, 6))

summary_data = []
for stat in stat_indices:
    for ch in correlation_df['channel'].unique():
        # Filter data
        filtered = correlation_df[(correlation_df['statistic'] == stat) &
                                  (correlation_df['channel'] == ch)]

        # Calculate mean absolute correlation and count significant correlations
        filtered_copy = filtered.copy()
        filtered_copy['r'] = abs(filtered_copy['r'])
        mean_abs_r = filtered_copy['r'].mean()
        sig_count = filtered['significant'].sum()
        total_count = len(filtered)

        summary_data.append({
            'statistic': stat,
            'channel': ch,
            'mean_abs_r': mean_abs_r,
            'sig_pct': sig_count / total_count * 100 if total_count > 0 else 0
        })

summary_df = pd.DataFrame(summary_data)

# Create heatmap of mean absolute correlation values
pivot_table = summary_df.pivot(index='statistic', columns='channel', values='mean_abs_r')
sns.heatmap(pivot_table, annot=True, cmap='Reds', fmt='.2f')
plt.title('Mean Absolute Correlation between Statistical Measures and TLX by Channel')
plt.tight_layout()
# plt.savefig('abs_correlation_heatmap.png', dpi=300)
plt.show()