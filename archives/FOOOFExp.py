import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, iirnotch, filtfilt
from fooof import FOOOF
import os
from itertools import product
from scipy.optimize import minimize
import random


def apply_notch_filters(data, fs, freqs, quality_factor=30):
    """
    Apply notch filters to the data at specified frequencies.

    Parameters:
    - data: ndarray, shape (n_samples, n_channels)
    - fs: float, sampling frequency
    - freqs: list of float, frequencies to notch out
    - quality_factor: float, quality factor for the notch filter

    Returns:
    - filtered_data: ndarray, filtered data
    """
    filtered_data = data.copy()
    for freq in freqs:
        b, a = iirnotch(freq / (fs / 2), quality_factor)
        filtered_data = filtfilt(b, a, filtered_data, axis=0)
    return filtered_data


def optimize_fooof_params_per_channel(emg_time_series, fs, n_channels, freq_range, nperseg, noverlap):
    """
    Optimize FOOOF parameters for each channel using randomly selected windows.

    Returns:
    - channel_params: dict, optimized FOOOF parameters per channel
    - channel_ap_fits: dict, aperiodic fits per channel
    """
    channel_params = {}
    channel_ap_fits = {}
    n_training_windows = 10  # Number of windows to use for training

    for channel_idx in range(n_channels):
        # Randomly select windows
        total_samples = emg_time_series.shape[0]
        window_length_samples = nperseg
        window_starts = np.arange(0, total_samples - window_length_samples + 1, window_length_samples)
        selected_windows = random.sample(list(window_starts), n_training_windows)

        psd_list = []
        for start_idx in selected_windows:
            end_idx = start_idx + window_length_samples
            channel_data = emg_time_series[start_idx:end_idx, channel_idx]

            # Compute PSD
            freqs, psd = welch(
                channel_data,
                fs=fs,
                nperseg=nperseg,
                noverlap=noverlap,
                scaling='density'
            )

            # Exclude frequencies outside freq_range
            freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
            freqs_used = freqs[freq_mask]
            psd_used = psd[freq_mask]

            psd_list.append(psd_used)

        # Average PSD across windows
        avg_psd = np.mean(psd_list, axis=0)

        # Optimize FOOOF parameters on the average PSD
        initial_params = [1, 12, 0.1, 5]
        bounds = [(0.5, 20), (1, 50), (0.01, 1.0), (1, 20)]

        result = minimize(
            fooof_objective,
            initial_params,
            args=(freqs_used, avg_psd, freq_range),
            bounds=bounds,
            method='L-BFGS-B'
        )

        if result.success:
            pwl_min_opt, pwl_max_opt, mph_opt, mnp_opt = result.x
            mnp_opt = int(round(mnp_opt))
            optimized_params = {
                'peak_width_limits': (pwl_min_opt, pwl_max_opt),
                'min_peak_height': mph_opt,
                'max_n_peaks': mnp_opt,
                'aperiodic_mode': 'knee',
                'verbose': False
            }
        else:
            print(f"Optimization failed for Channel {channel_idx}. Using default parameters.")
            optimized_params = {
                'peak_width_limits': (1, 12),
                'min_peak_height': 0.1,
                'max_n_peaks': 5,
                'aperiodic_mode': 'knee',
                'verbose': False
            }

        # Fit FOOOF on the average PSD with optimized parameters
        fm = FOOOF(**optimized_params)
        fm.fit(freqs_used, avg_psd, freq_range)

        # Save optimized parameters and aperiodic fit
        channel_params[channel_idx] = optimized_params
        channel_ap_fits[channel_idx] = fm.ap_fit_

    return channel_params, channel_ap_fits


def test_aperiodic_trend(emg_time_series, fs, channel_ap_fits, freq_range, n_channels, nperseg, noverlap):
    """
    Test the aperiodic trend on validation windows.

    Returns:
    - validation_results: dict, validation metrics per channel
    """
    validation_results = {}
    n_validation_windows = 5  # Number of windows to use for validation

    for channel_idx in range(n_channels):
        # Randomly select validation windows
        total_samples = emg_time_series.shape[0]
        window_length_samples = nperseg
        window_starts = np.arange(0, total_samples - window_length_samples + 1, window_length_samples)
        selected_windows = random.sample(list(window_starts), n_validation_windows)

        r_squared_list = []
        error_list = []

        for start_idx in selected_windows:
            end_idx = start_idx + window_length_samples
            channel_data = emg_time_series[start_idx:end_idx, channel_idx]

            # Compute PSD
            freqs, psd = welch(
                channel_data,
                fs=fs,
                nperseg=nperseg,
                noverlap=noverlap,
                scaling='density'
            )

            # Exclude frequencies outside freq_range
            freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
            freqs_used = freqs[freq_mask]
            psd_used = psd[freq_mask]

            # Compute log power
            log_psd = np.log10(psd_used)

            # Subtract aperiodic fit
            ap_fit = channel_ap_fits[channel_idx]
            periodic_log_psd = log_psd - ap_fit

            # Compute fit quality metrics
            residual = periodic_log_psd
            ss_res = np.sum(residual ** 2)
            ss_tot = np.sum((log_psd - np.mean(log_psd)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            error = np.sqrt(ss_res / len(residual))

            r_squared_list.append(r_squared)
            error_list.append(error)

        # Compute average metrics
        avg_r_squared = np.mean(r_squared_list)
        avg_error = np.mean(error_list)

        validation_results[channel_idx] = {
            'avg_r_squared': avg_r_squared,
            'avg_error': avg_error
        }

    return validation_results


def apply_FOOOF(data_obj):
    fs = 250  # Sampling frequency
    n_channels = 16  # Number of channels
    freq_range = [2, fs / 2]

    # Access the raw EMG data
    emg_stream = data_obj.ElectrodeStream
    emg_time_series = emg_stream['time_series']
    emg_time_stamps = emg_stream['time_stamps']

    # Apply notch filters
    emg_time_series = apply_notch_filters(emg_time_series, fs, freqs=[50, 100], quality_factor=30)

    # Adjust nperseg and noverlap
    nperseg = 1024
    noverlap = nperseg // 2

    # Step 1: Optimize FOOOF parameters per channel
    channel_params, channel_ap_fits = optimize_fooof_params_per_channel(
        emg_time_series, fs, n_channels, freq_range, nperseg, noverlap
    )

    # Step 2: Test the aperiodic trend
    validation_results = test_aperiodic_trend(
        emg_time_series, fs, channel_ap_fits, freq_range, n_channels, nperseg, noverlap
    )

    # Decision Rule: Check if avg_r_squared > threshold (e.g., 0.9)
    for channel_idx, metrics in validation_results.items():
        if metrics['avg_r_squared'] < 0.9:
            print(f"Channel {channel_idx} did not pass validation. Consider re-optimizing.")
        else:
            print(f"Channel {channel_idx} passed validation.")

    # Proceed to apply the trend to all windows
    # ... [Process each maze period and window as before]
    # Use the saved channel_params and channel_ap_fits to subtract the aperiodic component

    # Initialize lists to store results
    fooof_results = []

    # [Rest of your apply_FOOOF function, modifying it to use channel_ap_fits]

    # Return the results
    return fooof_results


def fooof_objective(params, freqs, psd, freq_range):
    pwl_min, pwl_max, mph, mnp = params
    mnp = int(round(mnp))  # Ensure integer value for max_n_peaks
    if pwl_min >= pwl_max:
        return np.inf  # Invalid parameter combination
    fm = FOOOF(
        peak_width_limits=(pwl_min, pwl_max),
        min_peak_height=mph,
        max_n_peaks=mnp,
        aperiodic_mode='knee',
        verbose=False
    )
    try:
        fm.fit(freqs, psd, freq_range)
        return fm.error_
    except Exception:
        return np.inf  # Return a large number if fitting fails


# def apply_FOOOF(data_obj):
#     fs = 250  # Sampling frequency
#
#     # Access the raw EMG data
#     emg_stream = data_obj.ElectrodeStream  # Replace with your actual EMG stream name
#     emg_time_series = emg_stream['time_series']  # Shape: (n_samples, n_channels)
#     emg_time_stamps = emg_stream['time_stamps']  # Shape: (n_samples,)
#
#     # Apply notch filters to remove line noise at 50 Hz and 100 Hz
#     emg_time_series = apply_notch_filters(emg_time_series, fs, freqs=[50, 100], quality_factor=30)
#
#     # Access trigger times and values
#     trigger_times = data_obj.Trigger_Cog['time_stamps']
#     trigger_values = data_obj.Trigger_Cog['time_series'][:, 0]  # Adjust indexing if needed
#
#     # Define the trigger codes for maze start and end
#     START_MAZE_TRIGGER = 4  # Replace with your actual start trigger code
#     END_MAZE_TRIGGER = 9    # Replace with your actual end trigger code
#
#     # Segment the data into maze-solving periods
#     maze_periods = []
#     is_in_maze = False
#     for i, trigger_value in enumerate(trigger_values):
#         if trigger_value == START_MAZE_TRIGGER:
#             start_time = trigger_times[i]
#             is_in_maze = True
#         elif trigger_value == END_MAZE_TRIGGER and is_in_maze:
#             end_time = trigger_times[i]
#             maze_periods.append((start_time, end_time))
#             is_in_maze = False
#
#     # Initialize lists to store results
#     fooof_results = []
#
#     # Process each maze period
#     for period_idx, (start_time, end_time) in enumerate(maze_periods):
#         # Find indices corresponding to the current maze period
#         indices = np.where((emg_time_stamps >= start_time) & (emg_time_stamps <= end_time))[0]
#         if len(indices) == 0:
#             continue  # Skip if no data in this period
#
#         # Extract the EMG segment for the current maze period
#         emg_segment = emg_time_series[indices, :]  # Shape: (n_samples_in_period, n_channels)
#
#         # Adjust nperseg based on the length of the data segment
#         segment_length = emg_segment.shape[0]
#         # Ensure nperseg is less than or equal to the segment length
#         nperseg = min(1024, segment_length)
#         # Adjust nperseg to the next power of 2 for efficiency
#         nperseg = 2 ** int(np.floor(np.log2(nperseg)))
#         # Ensure nperseg is at least a minimum length (e.g., 256 samples)
#         nperseg = max(nperseg, 256)
#         noverlap = nperseg // 2  # 50% overlap
#
#         # Optionally, segment the maze period into fixed-length windows
#         window_length_sec = 10  # Window length in seconds
#         window_length_samples = int(window_length_sec * fs)
#         if segment_length > window_length_samples:
#             # Split into overlapping windows
#             step = window_length_samples // 2  # 50% overlap
#             window_starts = np.arange(0, segment_length - window_length_samples + 1, step)
#         else:
#             # Use the entire segment as one window
#             window_starts = [0]
#             window_length_samples = segment_length
#
#         # Process each window within the maze period
#         for window_idx, start_idx in enumerate(window_starts):
#             end_idx = start_idx + window_length_samples
#             emg_window = emg_segment[start_idx:end_idx, :]  # Shape: (window_length_samples, n_channels)
#
#             # Compute power spectrum for each channel
#             for channel_idx in range(emg_window.shape[1]):
#                 channel_data = emg_window[:, channel_idx]
#
#                 # Compute power spectrum using Welch's method
#                 freqs, psd = welch(
#                     channel_data,
#                     fs=fs,
#                     nperseg=nperseg,
#                     noverlap=noverlap,
#                     scaling='density'
#                 )
#
#                 # Exclude 0 Hz and frequencies below 2 Hz
#                 freq_range = [2, fs / 2]
#                 freq_mask = freqs >= freq_range[0]
#                 freqs = freqs[freq_mask]
#                 psd = psd[freq_mask]
#
#                 # Compute log power of the PSD
#                 log_psd = np.log10(psd)
#
#                 # Define initial parameters for FOOOF
#                 initial_params = [2, 8, 0.1, 6]  # [pwl_min, pwl_max, mph, mnp]
#                 bounds = [(1, 3), (6, 10), (0.01, 0.5), (1, 10)]
#
#
#                 # Perform optimization
#                 result = minimize(
#                     fooof_objective,
#                     initial_params,
#                     args=(freqs, psd, freq_range),
#                     bounds=bounds,
#                     method='L-BFGS-B'
#                 )
#
#                 # Check if optimization was successful
#                 if result.success:
#                     # Extract optimized parameters
#                     pwl_min_opt, pwl_max_opt, mph_opt, mnp_opt = result.x
#                     mnp_opt = int(round(mnp_opt))
#                 else:
#                     print(f"Optimization failed for Channel {channel_idx}, Maze {period_idx}, Window {window_idx}. "
#                           f"Using initial parameters.")
#                     pwl_min_opt, pwl_max_opt, mph_opt, mnp_opt = initial_params
#                     mnp_opt = int(round(mnp_opt))
#
#                 # Fit FOOOF with optimized parameters
#                 fm = FOOOF(
#                     peak_width_limits=(pwl_min_opt, pwl_max_opt),
#                     min_peak_height=mph_opt,
#                     max_n_peaks=mnp_opt,
#                     aperiodic_mode='knee',  # or 'fixed'
#                     verbose=False
#                 )
#                 fm.fit(freqs, psd, freq_range)
#
#                 # Extract the periodic component
#                 # Compute log power of the PSD
#                 log_psd = np.log10(psd)
#
#                 # Subtract aperiodic fit to get periodic component in log space
#                 periodic_log_psd = log_psd - fm._ap_fit
#
#                 # Convert back to linear units
#                 periodic_psd = 10 ** periodic_log_psd
#
#                 # periodic_psd = 10 ** fm._peak_fit
#                 ap_fit_psd = 10 ** fm._ap_fit
#                 model_psd = 10 ** fm.fooofed_spectrum_
#
#
#                 # Store the results
#                 result = {
#                     'period_idx': period_idx,
#                     'window_idx': window_idx,
#                     'channel_idx': channel_idx,
#                     'aperiodic_params': fm.aperiodic_params_,
#                     'peak_params': fm.peak_params_,
#                     'r_squared': fm.r_squared_,
#                     'error': fm.error_,
#                     'freqs': freqs,
#                     'psd': psd,
#                     'aperiodic_fit': fm._ap_fit,
#                     'periodic_psd': periodic_psd,
#                     'fooofed_spectrum': fm.fooofed_spectrum_,
#                 }
#                 fooof_results.append(result)
#
#                 # Optional: Visualize and save the results
#                 output_dir = os.path.join(data_obj.output_folder, f'fooof_plots')
#                 os.makedirs(output_dir, exist_ok=True)
#                 fm.plot(title=f'Channel {channel_idx}, Maze {period_idx}, Window {window_idx}')
#                 plot_filename = f'fooof_channel_{channel_idx}_maze_{period_idx}_window_{window_idx}.png'
#                 # add r_squared and error inside the plot on the bottom left corner
#                 plt.text(0.1, 0.1, f'R^2: {fm.r_squared_:.2f}\nError: {fm.error_:.2f}', fontsize=12, transform=plt.gcf().transFigure)
#
#                 plt.savefig(os.path.join(output_dir, plot_filename))
#                 plt.close()
#
#                 # Optional: Plot the periodic component
#                 plt.figure()
#                 plt.plot(fm.freqs, psd, label='Original PSD', color='blue')
#                 plt.plot(fm.freqs, periodic_psd, label='Periodic PSD (Residual)', color='orange')
#                 plt.xlabel('Frequency (Hz)')
#                 plt.ylabel('Power (log scale)')
#                 plt.yscale('log')
#                 plt.title(f'Original and Periodic PSD - Channel {channel_idx}, Maze {period_idx}, Window {window_idx}')
#                 plt.legend()
#                 plot_filename = f'combined_psd_channel_{channel_idx}_maze_{period_idx}_window_{window_idx}.png'
#                 plt.savefig(os.path.join(output_dir, plot_filename))
#                 plt.close()
#
#                 # Plot all components
#                 plt.figure()
#                 plt.plot(freqs, psd, label='Original PSD', color='blue')
#                 plt.plot(freqs, ap_fit_psd, label='Aperiodic Fit', color='red')
#                 plt.plot(freqs, periodic_psd, label='Periodic PSD', color='orange')
#                 plt.plot(freqs, model_psd, label='Full Model Fit', color='green')
#                 plt.xlabel('Frequency (Hz)')
#                 plt.ylabel('Power')
#                 plt.yscale('log')
#                 plt.title(f'PSD and Model Components - Channel {channel_idx}, Maze {period_idx}, Window {window_idx}')
#                 plt.legend()
#                 plot_filename = f'psd_and_components_channel_{channel_idx}_maze_{period_idx}_window_{window_idx}.png'
#                 plt.savefig(os.path.join(output_dir, plot_filename))
#                 plt.close()
#
#         # You may choose to combine the periodic_psd across windows if desired
#
#     # Return the results
#     return fooof_results
