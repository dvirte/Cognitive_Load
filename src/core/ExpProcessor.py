import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from astropy.units.quantity_helper.function_helpers import block
from scipy.signal import spectrogram
from fooof import FOOOF
from scipy.signal import iirnotch, filtfilt
import warnings
from scipy.signal import welch
import math
from src.signal_processing.classifying_ica_components.classifying_ica_components import (perform_ica_algorithm,
                                                                   classify_participant_components_using_atlas,
                                                                   extract_and_order_ica_data)




class ExpProcessor:
    def __init__(self, emg_data, trigger_stream, fs=250, window_size=30.0, overlap=0.5, path=None,
                 auto_process=True, subject_id=None, sorted_indices=None):
        """
        Initializes the ExpProcessor with EMG data, triggers, and sampling frequency.
        Applies initial bandpass and notch filters.

        :param emg_data: Dict containing 'time_series' and 'time_stamps' for EMG data.
        :param trigger_stream: Dict containing 'time_series' and 'time_stamps' for triggers.
        :param fs: Sampling frequency in Hz.
        :param window_size: Size of each window in seconds for FOOOF analysis.
        :param overlap: Fraction of overlap between windows (e.g., 0.5 for 50% overlap).
        :param auto_process: If True, automatically process all periods upon initialization.
        :param subject_id: Identifier for the subject.
        :param sorted_indices: Array-like sorted indices based on cognitive load.
        """
        self.fs = fs
        self.data = emg_data['time_series']
        self.data_timestamps = emg_data['time_stamps']
        self.triggers = trigger_stream['time_series']
        self.triggers_time_stamps = trigger_stream['time_stamps']
        self.window_size = window_size
        self.overlap = overlap
        self.subject_id = subject_id
        self.sorted_indices = sorted_indices
        self.features_df = None

        # Apply initial filters
        self.apply_initial_filters()

        # Identify experimental periods
        self.rest_periods, self.play_periods, self.calibrate_periods = self.find_periods()

        # Initialize FOOOF object
        self.fooofer = FOOOF(peak_width_limits=[4, 10], max_n_peaks=6, min_peak_height=0.1)

        # Initialize storage for FOOOF results
        self.fooof_results = {}

        # Initialize storage for spectral features as a nested dictionary
        self.spectral_features = {
            "rest": [],
            "play": [],
            "calibrate": []
        }

        if auto_process:
            self.process_all_periods()

        # # Aplly ICA to the data
        # atlas_folder = r"classifying_ica_components\atlas"
        # self.im_dir = 'face_ica.jpeg'  # path to the image
        # x_coor_path = f"{atlas_folder}\side_x_coor.npy"
        # x_coor = np.load(x_coor_path)  # load x coordinates
        # y_coor_path = f"{atlas_folder}\side_y_coor.npy"
        # y_coor = np.load(y_coor_path)  # load y coordinates
        # self.wavelet = 'db15'  # wavelet type
        # threshold = np.load(f"{atlas_folder}/threshold.npy")
        # centroids_lst = []
        # match = re.search(r'data\\(participant_\d+)\\(S\d+)\\(\d+)\.xdf', path)
        # participant_ID = match.group(1)
        # session_number = match.group(2)
        # filename = match.group(3)
        # session_folder_path = f"data/{participant_ID}/{session_number}"
        # file_path = f"data/{participant_ID}/{session_number}/{filename}.xdf"
        # data_path = path[:path.index("data") + len("data")]
        #
        # for i in range(17):
        #     current_centroid = np.load(f"{atlas_folder}/cluster_{i + 1}.npy")
        #     current_centroid = np.nan_to_num(current_centroid, nan=0)
        #     centroids_lst.append(current_centroid)
        # experinment_part_name = ""  # name of the experiment part
        # if not os.path.exists(
        #         fr'{session_folder_path}\{participant_ID}_{session_number}_heatmap_{self.wavelet}.npy'):
        #     perform_ica_algorithm(file_path, participant_ID, session_number, session_folder_path,
        #                           self.im_dir, x_coor, y_coor, 16, fs=self.fs, data_xdf=self.data)
        # participant_data = np.load(
        #     fr'{session_folder_path}\{participant_ID}_{session_number}_heatmap_{self.wavelet}.npy')
        # participant_data = np.nan_to_num(participant_data, nan=0)
        # classify_participant_components_using_atlas(participant_data, data_path, centroids_lst,
        #                                             participant_ID, session_folder_path,
        #                                             session_number,
        #                                             threshold, self.wavelet, experinment_part_name,
        #                                             self.im_dir, participant_ID, session_folder_path,
        #                                             x_coor, y_coor)
        # self.data = extract_and_order_ica_data(participant_ID, session_folder_path, session_number)
        # self.data = np.transpose(self.data)

    def apply_initial_filters(self):
        """
        Applies initial notch filters to the data.
        """
        # Apply notch filters to remove powerline noise (50Hz and its harmonics)
        self.apply_notch_filters([50, 100], quality_factor=30)

    def apply_notch_filters(self, freqs, quality_factor=30):
        """
        Apply multiple notch filters to the data to remove specified frequencies.

        :param freqs: List of frequencies to notch out.
        :param quality_factor: Quality factor for the notch filters.
        """
        for freq in freqs:
            b, a = iirnotch(freq, quality_factor, self.fs)
            self.data = filtfilt(b, a, self.data, axis=0)

    def find_periods(self):
        """
        Identifies rest, play, and calibration periods based on triggers.
        Returns lists of (start_time, end_time) tuples for each period.
        """
        rest_periods = []
        play_periods = []
        calibrate_periods = []
        last_time = self.data_timestamps[-1]  # Last timestamp in your data

        triggers = self.triggers
        trigger_times = self.triggers_time_stamps

        # Initialize variables
        cal_start_time = None
        play_start_time = None
        i = 0

        while i < len(triggers):
            trigger = triggers[i]
            trigger_time = trigger_times[i]

            # Calibration periods
            if (13 <= trigger <= 18) or (21 <= trigger <= 27):
                cal_start_time = trigger_time
                # Calibration ends when a trigger outside calibration range occurs
                j = i + 1
                while j < len(triggers):
                    next_trigger = triggers[j]
                    if not ((13 <= next_trigger <= 18) or (21 <= next_trigger <= 27)):
                        break
                    j += 1
                if j < len(triggers):
                    cal_end_time = trigger_times[j]
                else:
                    cal_end_time = last_time
                calibrate_periods.append((cal_start_time, cal_end_time))
                i = j
                continue

            # Play periods
            elif trigger == 4:
                # Check if the previous trigger was 20
                if i > 0 and triggers[i - 1] == 20:
                    # This is a new maze within the same stage, so continue without marking a new play period
                    i += 1
                    continue
                else:
                    play_start_time = trigger_time
                    # Play ends with trigger 11 (NASA-TLX rating) or 5 (end of experiment)
                    j = i + 1
                    while j < len(triggers):
                        next_trigger = triggers[j]
                        next_trigger_time = trigger_times[j]
                        if next_trigger == 11 or next_trigger == 5:
                            break
                        elif next_trigger == 4 and triggers[j - 1] == 20:
                            # Skip over the 4 after 20, it's the same play period
                            j += 1
                            continue
                        j += 1
                    if j < len(triggers):
                        play_end_time = trigger_times[j]
                    else:
                        play_end_time = last_time
                    play_periods.append((play_start_time, play_end_time))
                    i = j
                    continue

            i += 1

        # Calculate rest periods as the gaps between play and calibration periods
        # Merge play and calibration periods
        combined_periods = play_periods + calibrate_periods
        combined_periods.sort(key=lambda x: x[0])  # Sort by start time

        # Find rest periods between combined periods
        prev_end = self.data_timestamps[0]  # Start from the beginning of the data
        for period in combined_periods:
            start, end = period
            if start > prev_end:
                rest_periods.append((prev_end, start))
            prev_end = max(prev_end, end)
        # Add rest period after the last period if necessary
        if prev_end < last_time:
            rest_periods.append((prev_end, last_time))

        return rest_periods, play_periods, calibrate_periods

    def segment_data(self, data, start_time, end_time):
        """
        Extracts data segments based on start and end times.

        :param data: Numpy array of data.
        :param start_time: Start time in seconds.
        :param end_time: End time in seconds.
        :return: Tuple of (data_segment, timestamps_segment)
        """
        start_idx = np.searchsorted(self.data_timestamps, start_time)
        end_idx = np.searchsorted(self.data_timestamps, end_time, side='right')
        return data[start_idx:end_idx, :], self.data_timestamps[start_idx:end_idx]

    def window_data(self, data_segment, window_size, overlap):
        """
        Divides a data segment into overlapping windows.

        :param data_segment: Numpy array of data for a specific period.
        :param window_size: Size of each window in seconds.
        :param overlap: Fraction of overlap between windows.
        :return: List of windows (each window is a numpy array).
        """
        window_size_samples = int(window_size * self.fs)
        step_size = int(window_size_samples * (1 - overlap))
        windows = []
        num_samples = data_segment.shape[0]

        start = 0
        while start + window_size_samples <= num_samples:
            window = data_segment[start:start + window_size_samples, :]
            windows.append(window)
            start += step_size

        # Handle the last window
        if start < num_samples:
            window = data_segment[-window_size_samples:, :]  # Take the last window_size_samples
            windows.append(window)

        return windows

    def extract_trials(self, status=1, period=0):
        """
        Extracts the EMG data for a specific level and response status.

        Parameters:
        status (int): 0 for rest, 1 for play, 2 for calibration.
        period (int): the specific period to extract the data from.
        kind (str): the type of data to extract (e.g., 'EEG_alpha', 'EEG_beta', 'EEG_theta', 'EEG_delta', 'EMG').

        Returns:
        ndarray: tuple[0] contains the EMG data for the specific level and response status.
        ndarray: tuple[1] contains the timestamps for the EMG data.
        ndarray: tuple[2] contains the triggers
        ndarray: tuple[3] contains the time stamps of the triggers
        tuple: tuple[4] contains the status, period and kind of the data
        """
        if status == 0:
            periods = self.rest_periods
        elif status == 1:
            periods = self.play_periods
        elif status == 2:
            periods = self.calibrate_periods
        else:
            raise ValueError("status must be 0, 1 or 2")

        if status != 2:
            start, stop = periods[period]
        else:
            start, stop = periods[0][0], periods[-1][1]


        # Find the indices of the time stamps that fall within the game time stamps
        start_idx = np.searchsorted(self.data_timestamps, start)
        if self.data_timestamps[start_idx] > start:
            start_idx = max(start_idx - 1, 0)
        stop_idx = np.searchsorted(self.data_timestamps, stop, side='right')
        if self.data_timestamps[min(stop_idx, len(self.data_timestamps) - 1)] < stop:
            stop_idx = min(stop_idx + 1, len(self.data_timestamps))
        mask = ((self.triggers_time_stamps >= self.data_timestamps[start_idx]) &
                (self.triggers_time_stamps <= self.data_timestamps[min(stop_idx, len(self.data_timestamps) - 1)]))
        indices = np.where(mask)[0]

        if status == 1:
            if self.triggers[indices[0]] == 8:
                indices = np.delete(indices, 0)
            if self.triggers[indices[-1]] == 11:
                indices = np.delete(indices, -1)

        if status == 0:
            if self.triggers[indices[0]] == 9:
                indices = np.delete(indices, 0)
            if self.triggers[indices[-1]] == 4:
                indices = np.delete(indices, -1)

        return (self.data[start_idx:stop_idx, :], self.data_timestamps[start_idx:stop_idx],
                self.triggers[indices], self.triggers_time_stamps[indices], (status, period))

    def apply_fooof_to_window(self, window_data, period_type, period_idx, window_idx):
        """
        Applies FOOOF to each channel in a window and extracts spectral features.

        :param window_data: Numpy array of data in the window (samples x channels).
        :param period_type: Type of period ('rest', 'play', 'calibrate').
        :param period_idx: Index of the period within its type.
        :param window_idx: Index of the window within the period.
        """
        self.spectral_features[period_type][period_idx].append([])

        for channel_idx in range(window_data.shape[1]):
            channel_data = window_data[:, channel_idx]
            # Compute power spectral density (PSD) using Welch's method
            freqs, psd = self.compute_psd(channel_data)
            # Fit FOOOF model
            try:
                self.fooofer.fit(freqs, psd)
                # Extract features
                aperiodic_params = self.fooofer.aperiodic_params_
                peak_params = self.fooofer.peak_params_
                # Store results
                feature = {
                    'channel': channel_idx,
                    'aperiodic_offset': aperiodic_params[0],
                    'aperiodic_slope': aperiodic_params[1],
                    'num_peaks': len(peak_params),
                    'peak_params': peak_params,
                    'freqs': freqs,
                    'psd': psd,
                    'channel_data': channel_data,
                }
                self.spectral_features[period_type][period_idx][window_idx].append(feature)
            except Exception as e:
                warnings.warn(
                    f"FOOOF failed for channel {channel_idx} in {period_type} period {period_idx} window {window_idx}: {e}")
                continue

    def compute_psd(self, channel_data):
        """
        Computes the Power Spectral Density (PSD) using Welch's method.

        :param channel_data: Numpy array of channel data.
        :return: Tuple of (frequencies, PSD values)
        """
        freqs, psd = welch(
            channel_data,
            fs=self.fs,
            nperseg=min(len(channel_data), 125),
            noverlap=min(round(len(channel_data) / 2), 62),
            scaling='density'
        )

        # Remove 0 Hz component to prevent FOOOF from processing it
        if freqs[0] == 0:
            freqs = freqs[1:]
            psd = psd[1:]

        return freqs, psd

    def process_all_periods(self):
        """
        Processes all experimental periods by segmenting, windowing, and applying FOOOF.
        """
        periods = {
            'rest': self.rest_periods,
            'play': self.play_periods,
            'calibrate': self.calibrate_periods
        }

        for period_type, period_list in periods.items():
            for idx, (start_time, end_time ) in enumerate(period_list):
                self.spectral_features[period_type].append([])  # Initialize list for this period
                duration = (end_time  - start_time)  # Duration in seconds
                # Determine window size
                if duration < self.window_size:
                    window_size = duration
                    overlap = 0  # No overlap needed
                else:
                    window_size = self.window_size
                    overlap = self.overlap

                data_segment, _ = self.segment_data(self.data, start_time, end_time)
                windows = self.window_data(data_segment, window_size, overlap)
                for window_idx, window in enumerate(windows):
                    self.apply_fooof_to_window(window, period_type, idx, window_idx)
                    # Optionally, store window-specific metadata if needed

        # After processing all periods, extract features
        self.features_df = self.extract_features_dataframe()

    def extract_features_dataframe(self):
        """
        Converts the nested spectral features dictionary into a Pandas DataFrame.

        :return: Pandas DataFrame containing spectral features.
        """
        records = []
        for period_type, periods in self.spectral_features.items():
            for period_idx, windows in enumerate(periods):
                for window_idx, window in enumerate(windows):
                    for feature in window:
                        record = {
                            'period_type': period_type,
                            'period_idx': period_idx,
                            'window_idx': window_idx,
                            'channel': feature['channel'],
                            'aperiodic_offset': feature['aperiodic_offset'],
                            'aperiodic_slope': feature['aperiodic_slope'],
                            'num_peaks': feature['num_peaks']
                        }
                        # Include peak frequencies and amplitudes
                        for peak_num, peak in enumerate(feature['peak_params'], start=1):
                            freq = peak[0]
                            amp = peak[1]
                            record[f'peak_{peak_num}_freq_Hz'] = freq
                            record[f'peak_{peak_num}_amp'] = amp
                        records.append(record)
        return pd.DataFrame(records)

    def load_features(self, filepath):
        """
        Loads spectral features from a CSV file into the class.

        :param filepath: Path to the CSV file.
        """
        try:
            self.features_df = pd.read_csv(filepath)
            print(f"Spectral features loaded from {filepath}")
        except FileNotFoundError:
            warnings.warn(f"File not found: {filepath}")
        except Exception as e:
            warnings.warn(f"Failed to load features from {filepath}: {e}")

    def plot_fooof(self, top_n_windows=6, indx=0, least_tow = False):
        # Retrieve the most cognitively loaded trial based on the sorted indices
        complex_trial = self.sorted_indices[indx]
        time_maze = self.play_periods[complex_trial][1] - self.play_periods[complex_trial][0]

        while True:
            if time_maze < 30:
                print(f'Maze {complex_trial + 1} is less than 30 seconds long. Moving to the next maze.')
                indx -= 1
                complex_trial = self.sorted_indices[indx]
                time_maze = self.play_periods[complex_trial][1] - self.play_periods[complex_trial][0]
                continue

            complex_trial_windows = self.spectral_features['play'][complex_trial]

            if least_tow:
                if len(complex_trial_windows) < 2:
                    print(f'Maze {complex_trial + 1} has less than 2 windows. Moving to the next maze.')
                    indx -= 1
                    complex_trial = self.sorted_indices[indx]
                    time_maze = self.play_periods[complex_trial][1] - self.play_periods[complex_trial][0]
                    continue
            # Determine the indices for the selected windows

            print(f'Processing maze {complex_trial + 1} with {len(complex_trial_windows)} windows')
            break

        if len(complex_trial_windows) < top_n_windows:
            selected_window_indices = list(
                range(len(complex_trial_windows)))  # Use all windows if less than top_n_windows
        else:
            # Choose `top_n_windows` windows at equal intervals within the trial
            interval = len(complex_trial_windows) / top_n_windows
            selected_window_indices = np.round([i * interval for i in range(top_n_windows)]).astype(int)

        # Loop over each electrode (16 total) and create a figure with subplots for each electrode
        for electrode in range(16):
            num_windows = len(selected_window_indices)
            if num_windows == 1:
                # Special case for only one window
                fig, ax = plt.subplots(figsize=(8, 6))
                fig.suptitle(f'Electrode {electrode + 1} - Maze no. {complex_trial + 1}')
                feature = complex_trial_windows[selected_window_indices[0]][electrode]
                freqs, psd = feature['freqs'], feature['psd']

                # Apply FOOOF to the single window
                self.fooofer.fit(freqs, psd)

                # Extract FOOOF's results
                log_psd = np.log10(psd)  # Convert the raw PSD to log scale
                log_aperiodic_fit = self.fooofer._ap_fit  # Aperiodic fit is already in log scale
                flattened_spectrum = self.fooofer.power_spectrum - self.fooofer._ap_fit  # Flattened power spectrum

                # Plot each component
                ax.plot(freqs, log_psd, label='Log Raw Signal', color='black')
                ax.plot(freqs, log_aperiodic_fit, label='Log Aperiodic Fit', color='red', linestyle='--')
                ax.plot(freqs, flattened_spectrum, label='Flattened Spectrum (Aligned)', color='blue')
                ax.set_title('Single Window')
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Log Power')
                ax.legend()

            else:
                # General case for multiple windows
                num_rows = 2
                num_cols = math.ceil(num_windows / num_rows)  # Distribute subplots across 2 rows
                fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 8))
                fig.suptitle(f'Electrode {electrode + 1} - Maze no. {complex_trial + 1}')

                for subplot_idx, window_idx in enumerate(selected_window_indices):
                    # Get the row and column index for the current subplot
                    row = subplot_idx // num_cols
                    col = subplot_idx % num_cols

                    # Get the FOOOF feature data for the specific window and electrode
                    feature = complex_trial_windows[window_idx][electrode]
                    freqs, psd = feature['freqs'], feature['psd']

                    # Apply FOOOF to the current window
                    self.fooofer.fit(freqs, psd, freq_range=[1, 40])
                    if electrode > 11:
                        self.fooofer.plot(plot_peaks='shade', peak_kwargs={'color' : 'green'})
                        plt.title(f'Electrode {electrode + 1} - Maze no. {complex_trial + 1} - Window {window_idx + 1}')
                        err = self.fooofer.error_
                        R2 = self.fooofer.r_squared_
                        plt.text(0.5, 0.95, f'Error: {err:.2f} - R^2: {R2:.2f}', horizontalalignment='center',
                                    verticalalignment='top')
                        # save and close the plot
                        plt.savefig(f'fooof_plots/electrode_{electrode + 1}_maze_{complex_trial + 1}_window_{window_idx + 1}.png')
                        plt.close()

                    # Extract FOOOF's results
                    log_aperiodic_fit = self.fooofer._ap_fit  # Aperiodic fit is already in log scale
                    flattened_spectrum = self.fooofer.power_spectrum - self.fooofer._ap_fit  # Flattened power spectrum
                    log_psd = np.log10(psd)[0:len(log_aperiodic_fit)]  # Convert the raw PSD to log scale
                    freqs = freqs[0:len(log_aperiodic_fit)]

                    # Plot each component
                    if num_cols >1:
                        axs[row, col].plot(freqs, log_psd, label='Log Raw Signal', color='black')
                        axs[row, col].plot(freqs, log_aperiodic_fit, label='Log Aperiodic Fit', color='red', linestyle='--')
                        axs[row, col].plot(freqs, flattened_spectrum, label='Flattened Spectrum (Aligned)', color='blue')
                        axs[row, col].set_title(f'Window {window_idx + 1}')
                        axs[row, col].set_xlabel('Frequency (Hz)')
                        axs[row, col].set_ylabel('Log Power')
                        axs[row, col].legend()
                    else:
                        axs[row].plot(freqs, log_psd, label='Log Raw Signal', color='black')
                        axs[row].plot(freqs, log_aperiodic_fit, label='Log Aperiodic Fit', color='red', linestyle='--')
                        axs[row].plot(freqs, flattened_spectrum, label='Flattened Spectrum (Aligned)', color='blue')
                        axs[row].set_title(f'Window {window_idx + 1}')
                        axs[row].set_xlabel('Frequency (Hz)')
                        axs[row].set_ylabel('Log Power')
                        axs[row].legend()


                # Hide any unused subplots if there are fewer than 6 windows
                for remaining in range(subplot_idx + 1, num_rows * num_cols):
                    row = remaining // num_cols
                    col = remaining % num_cols
                    fig.delaxes(axs[row, col])

            # Display the plot for the current electrode
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)  # Adjust layout for the main title
            plt.show(block=False)
