import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
from scipy.signal import iirnotch, filtfilt
from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from sklearn.decomposition import FastICA


class ExpObj():
    def __init__(self, emg_data, trigger_stream):
        """
        Initializes the EEGDataProcessor with EEG data, triggers, and timestamps.

        :param eeg_data: The EEG data as a numpy array.
        :param triggers: List of trigger names (e.g., 'start rest', 'stop rest').
        :param time_stamps: List of timestamps corresponding to each trigger.
        """
        self.data = emg_data['time_series']
        self.data_timestamps = emg_data['time_stamps']
        self.triggers = trigger_stream['time_series']  # Assuming this is a list of triggers
        self.triggers_time_stamps = trigger_stream['time_stamps']
        self.rest_periods, self.play_periods, self.calibrate_periods = self.find_periods()
        self.fs = 250
        self.trials = None
        self.responses = None
        self.levels = None
        self.grouped_responses = None
        self.stim_start_idx = None
        self.aligned_correlated_trials = None
        self.trials_by_level = None

    def find_periods(self):
        """
        Identifies the rest periods based on triggers and their corresponding time stamps.
        """
        rest_periods = []
        play_periods = []
        calibrate_periods = []
        end_idx = None
        cal_ind = None

        for i, trigger in enumerate(self.triggers):
            if trigger == 6 or trigger == 9:
                start_idx = i
                if end_idx is not None:
                    play_periods.append((self.triggers_time_stamps[end_idx], self.triggers_time_stamps[start_idx]))
            elif trigger == 4 or trigger == 5:
                end_idx = i
                rest_periods.append((self.triggers_time_stamps[start_idx], self.triggers_time_stamps[end_idx]))
            elif 13 <= trigger <= 18:
                if cal_ind is not None:
                    calibrate_periods.append((self.triggers_time_stamps[cal_ind], self.triggers_time_stamps[i]))
                cal_ind = i

        if trigger != 5:
            rest_periods.append((self.triggers_time_stamps[end_idx], self.triggers_time_stamps[-1]))
        return rest_periods, play_periods, calibrate_periods

    def extract_emg_trials(self, status, period):
        """
        Extracts the EMG data for a specific level and response status.

        Parameters:
        status (int): 0 for rest, 1 for play.
        period (int): the specific period to extract the data from.

        Returns:
        list: list[0] contains the EMG data for the specific level and response status.
        list: list[1] contains the timestamps for the EMG data.
        """
        if status == 0:
            periods = self.rest_periods
        elif status == 1:
            periods = self.play_periods
        else:
            raise ValueError("status must be 0 or 1")

        start, stop = periods[period]

        # Find the indices of the time stamps that fall within the game time stamps
        start_idx = np.searchsorted(self.data_timestamps, start)
        stop_idx = np.searchsorted(self.data_timestamps, stop, side='right')
        return self.data[start_idx:stop_idx, :], self.data_timestamps[start_idx:stop_idx]

    def plot_data(self, trail):
        """
        Plots the EMG data on single plot.

        Parameters:
        trail (list): trail[0] contains the EMG data for the specific level and response status.
        trail (list): trail[1] contains the timestamps for the EMG data.

        Returns:
        None: The EMG data is plotted.
        """
        # convert to seconds
        time = trail[1] - trail[1][0]
        plt.figure()
        plt.plot(time, trail[0])
        plt.xlabel('Time (s)')
        plt.ylabel('EMG (mV)')
        plt.title('EMG Data')
        plt.show(block=False)
        input("Press Enter to keep going...")

    def frequency_spectrum(self, emg_data):
        """
        Plots the frequency spectrum of EMG data

        Parameters:
        emg_data

        Returns:
        None: The EMG data is plotted.
        """

        Fs = self.fs
        N = emg_data.shape[0]

        # Create a frequency axis
        freqs = np.fft.rfftfreq(N, 1 / Fs)

        # Loop through each channel
        for i in range(emg_data.shape[1]):
            # Compute the FFT
            fft_values = np.fft.rfft(emg_data[:, i])

            # Compute the magnitude spectrum
            mag_spectrum = np.abs(fft_values)

            # Plot the spectrum
            plt.figure(figsize=(10, 6))
            plt.plot(freqs, mag_spectrum)
            plt.title(f'Frequency Spectrum of EMG Channel {i + 1}')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            plt.xlim([0, Fs / 2])  # Show only up to the Nyquist frequency
            plt.grid(True)
            plt.show(block=False)
            input("Press Enter to keep going...")

    def plot_trail(self, trail):
        """
        Plots a series of subplots for each channel in the trail data.

        Parameters:
        - self: The object instance.
        - trail: A tuple where trail[0] is an ndarray of shape (samples, channels)
                 representing the data for each channel, and trail[1] is an ndarray
                 of shape (samples,) representing the timestamps.

        Returns:
        - None: The plot is displayed.
        """
        data, timestamps = trail
        timestamps = timestamps - timestamps[0]  # Convert to seconds
        samples, channels = data.shape

        fig, axs = plt.subplots(channels, 1, figsize=(10, 20), sharex=True)  # 16 rows, 1 col, shared X-axis

        for i in range(channels):
            axs[i].plot(timestamps, data[:, i])  # Plot each channel
            axs[i].text(1.02, 0.5, f'EMG {i + 1}', transform=axs[i].transAxes, va='center', ha='left')
            axs[i].spines['top'].set_visible(False)  # Hide the top spine
            axs[i].spines['right'].set_visible(False)  # Hide the right spine
            axs[i].spines['left'].set_visible(False)  # Optionally, hide the left spine as well
            axs[i].spines['bottom'].set_visible(False)  # Hide the bottom spine
            axs[i].get_xaxis().set_visible(False)  # Hide x-axis labels and ticks

        # Adjust settings for the last subplot
        axs[-1].spines['bottom'].set_visible(True)  # Show the bottom spine for the last subplot
        axs[-1].get_xaxis().set_visible(True)  # Show x-axis labels and ticks for the last subplot
        axs[-1].set_xlabel("Time [s]")  # Common X-axis title
        fig.text(0.04, 0.5, 'Amplitude [mV]', va='center', rotation='vertical')  # Common Y-axis title

        plt.subplots_adjust(hspace=0)  # Remove horizontal space between plots
        plt.show(block=False)
        input("Press Enter to keep going...")

    def create_trials(self, mode):

        trials = []
        responses = []
        levels = []
        grouped_responses = {}
        stim_start_idx = []
        trials_by_level = {}
        pre_trigger_offset = int(0.1 * self.fs)  # 0.5 seconds before trigger
        post_trigger_offset = int(1.5 * self.fs)  # 1 second after trigger

        for i, (trigger, timestamp) in enumerate(zip(self.triggers, self.triggers_time_stamps)):

            if 'play_stimulus' in trigger[0] and mode in trigger[0] and 'training' not in self.triggers[i + 2][
                0] and 'press' in self.triggers[i + 2][0]:
                trigger_index = np.searchsorted(self.data_timestamps, timestamp)
                start_idx = max(trigger_index - pre_trigger_offset, 0)
                end_idx = min(trigger_index + post_trigger_offset, len(self.data_timestamps))
                trial_data = self.data[start_idx:end_idx, :]
                trials.append(trial_data)
                response = 1 if 'return' in self.triggers[i + 2][0] else 0
                responses.append(response)
                parts = self.triggers[i + 2][0].split('_')
                # Assuming the level is at a specific position in the trigger string
                level = parts[2]
                # Adjust the index based on your trigger string format
                levels.append(level)
                stim_start_idx.append(start_idx)
                if level not in grouped_responses:
                    grouped_responses[level] = []
                grouped_responses[level].append(response)
                if level not in trials_by_level:
                    trials_by_level[level] = []
                trials_by_level[level].append(trial_data)

        level_keys = list(grouped_responses.keys())
        for level in level_keys:
            new_level = float(level)
            grouped_responses[new_level] = grouped_responses.pop(level)

        self.grouped_responses = dict(sorted(grouped_responses.items()))
        self.trials = trials
        self.responses = responses
        self.levels = levels
        self.stim_start_idx = stim_start_idx
        self.trials_by_level = trials_by_level

    def calculate_psychometric_curve(self, plot_figure=False):
        grouped_responses = self.grouped_responses
        levels = sorted(grouped_responses.keys())
        hit_rates = [np.mean(grouped_responses[level]) for level in levels]
        notfit = False
        initial_guess = [max(hit_rates), np.median(levels), 1, min(hit_rates)]

        def sigmoid_model(x, L, x0, k, b):
            try:
                return L / (1 + np.exp(-k * (x - x0))) + b
            except:
                return np.arange(0, len(x), 1)

        try:
            params, _ = curve_fit(sigmoid_model, levels, hit_rates, p0=initial_guess)

            # Find the 50% Level
            sigmoid_half = lambda x: sigmoid_model(x, *params) - 0.5

            level_50_percent = fsolve(sigmoid_half, np.median(levels))[0]

        except:
            notfit = True
        # Convert data points to PsychoPy coordinates
        # Assuming levels range from 0 to 1 and hit_rates range from 0 to 1
        fine_grained_levels = np.linspace(min(levels), max(levels), 500)
        if plot_figure:
            plt.figure()
            plt.plot(levels, hit_rates, 'o', label='Raw Data')
            if notfit:
                plt.plot(fine_grained_levels, sigmoid_model(fine_grained_levels, *params), label='Sigmoid Fit')

            plt.xlabel('Stimulation Level')
            plt.ylabel('Hit Rate')
            plt.title(f"Psychometric Curve")
            plt.legend()
            plt.grid(True)
            plt.show()

    def apply_notch_filters(self, freqs, quality_factor=30):
        """
        Apply multiple notch filters to the EMG signal and update the EMG_signal attribute.

        Parameters:
        freqs (list of float): Frequencies to be notched out.
        quality_factor (float): Quality factor of the notch filter.

        Returns:
        None: The EMG_signal attribute is updated in place.
        """
        for freq in freqs:
            b, a = iirnotch(freq, quality_factor, self.fs)
            self.data = filtfilt(b, a, self.data, axis=0)

    def dft_multichannel(self, nfft, axis=-1):
        f = np.fft.fftfreq(nfft, 1 / self.fs)
        f_keep = f > 0
        f = f[f_keep]

        n_channels = self.data.shape[-1]
        fft_values = np.zeros((n_channels, len(f)))

        for i in range(n_channels):
            y = np.abs(np.fft.fft(self.data[:, i], nfft, axis=axis))[f_keep]
            fft_values[i] = y

        return f, fft_values

    def average_chunks_by_mode(self, aligned_emf_chunks_by_mode):
        """
        Average the chunks for each stimulation mode.

        Parameters:
        aligned_emf_chunks_by_mode (dict): Dictionary containing aligned chunks, organized by stimulation modes.

        Returns:
        dict: Dictionary containing averaged chunks for each stimulation mode.
        """
        averaged_chunks_by_mode = {}
        for mode, chunks in aligned_emf_chunks_by_mode.items():
            # Ensuring all chunks have the same shape for averaging
            min_length = min(chunk['aligned_chunk'].shape[0] for chunk in chunks)
            aligned_chunks = [chunk['aligned_chunk'][:min_length, :] for chunk in chunks]

            # Calculating the average
            average_chunk = np.mean(aligned_chunks, axis=0)
            averaged_chunks_by_mode[mode] = average_chunk

        return averaged_chunks_by_mode

    def apply_bandpass_filter(self, lowcut, highcut, order=5):
        """
        Apply a bandpass Butterworth filter to the EMG signal.

        Parameters:
        lowcut (float): Low cut-off frequency of the filter.
        highcut (float): High cut-off frequency of the filter.
        order (int): Order of the filter.

        Returns:
        None: The EMG_signal attribute is updated in place.
        """
        nyq = 0.5 * self.fs  # Nyquist Frequency

        # Check if lowcut and highcut are within valid range
        if not 0 < lowcut < nyq:
            raise ValueError(f"Low cut-off frequency must be between 0 and {nyq}")
        if not 0 < highcut < nyq:
            raise ValueError(f"High cut-off frequency must be between 0 and {nyq}")

        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        self.data = filtfilt(b, a, self.data, axis=0)

    def get_recorded_pulses(self, win_size):
        """result is 3D numpy array: pulse num x channel x segment data"""
        win_size = int(win_size / 1000 * self.fs)
        data = np.array(self.trials)
        data = np.transpose(data, (0, 2, 1))
        result = []
        if len(data.shape) == 1:
            data = np.vstack([data, data])
        for idx in self.stim_start_idx:
            start = max(0, idx - win_size // 2)
            end = min(data.shape[1], idx + win_size // 2)
            result.append(data[:, start:end])
        result = np.array(result)
        return result

    def pulse_correlation(self, win_size=2, visualize=False):
        print('Align pulses...')
        # pulses_win = self.get_recorded_pulses(win_size)
        pulses_win = np.array(self.trials)
        pulses_win = np.transpose(pulses_win, (0, 2, 1))
        pulses_win_aligned = []
        for channel in range(pulses_win.shape[1]):
            data = pulses_win[:, channel, :]
            num_segments, samples_per_segment = data.shape
            synced_data = np.zeros((num_segments, samples_per_segment))
            synced_data[0, :] = data[0, :]
            # Iterate through pairs of segments to find time delays and synchronize
            for i in range(1, num_segments):
                corr = np.correlate(data[0], data[i], mode='full')
                max_idx = np.argmax(corr)
                delay = max_idx - samples_per_segment + 1

                # Shift segments based on calculated delay and pad with zeros
                if delay < 0:
                    synced_data[i, :delay] = data[i, -delay:]
                if delay > 0:
                    synced_data[i, delay:] = data[i, delay:]
                else:
                    synced_data[i, :] = data[i, :]
            pulses_win_aligned.append(synced_data)
        pulses_win_aligned = np.array(pulses_win_aligned).transpose(1, 0, 2)

        self.aligned_correlated_trials = pulses_win

    def clean_grouped_pulses_ica(self, pulses, visualize=True):
        """result is 3D numpy array: pulse num x channel x segment data"""

        artifacts_mat = []
        signals_mat = []

        for channel_ind in range(0, pulses.shape[1]):
            print(f'Processing channel {channel_ind}')
            data = pulses[:, channel_ind, :].T

            # # Check distribution
            # shapiro_test_result = shapiro(data[:, 0])
            # shapiro_p_value = shapiro_test_result[1]
            # alpha = 0.05
            # if shapiro_p_value > alpha:
            #     print(f'Channel {channel_ind} distribution is Gaussian')
            #     continue

            # Create artifact template
            art_template = np.mean(data, axis=1)

            # Preform ICA
            ica = FastICA(n_components=6, random_state=42, max_iter=100)
            ica.fit(data)
            print(f'Number of iterations taken for convergence: {ica.n_iter_}')
            components = ica.transform(data)

            # For each component, calculate correlation to the template
            pp_list = []
            corr_list = []
            pp_sign_list = []
            for ind in range(0, 6):
                # fig=plt.figure()
                # plt.plot(components[:, ind]-10*ind)
                corr = np.corrcoef(components[100:150, ind], art_template[100:150])[0, 1]
                corr_list.append(abs(corr))
                largest_peak_index = np.argmax(np.abs(components[:, ind]))
                pp_list.append(largest_peak_index)
                pp_sign_list.append(components[largest_peak_index, ind])
                # pp_list.append(np.max(components[:, ind]))

            # Divide components to artifact-related and signal-related
            corr_thresh = 0.1  # was 0.1
            pp_thresh = 160

            art_related_comp = np.where((np.array(corr_list) > corr_thresh) | (np.array(pp_list) < pp_thresh))
            # art_related_comp = np.where(
            #     (np.array(pp_sign_list) > 0) &
            #     ((np.array(corr_list) > corr_thresh) | (np.array(pp_list) < pp_thresh)))

            signal_comp = components.copy()
            signal_comp[:, art_related_comp] = 0
            restored_signal = ica.inverse_transform(signal_comp)
            signals_mat.append(restored_signal)

            signal_related_comp = np.setdiff1d(np.arange(0, 6), art_related_comp)
            art_comp = components.copy()
            art_comp[:, signal_related_comp] = 0
            restored_artifact = ica.inverse_transform(art_comp)
            artifacts_mat.append(restored_artifact)

            # plt.plot(np.array(corr_list)-channel_ind)

        signals_mat_3d = np.stack(signals_mat, axis=1).transpose(2, 1, 0)
        artifacts_mat_3d = np.stack(artifacts_mat, axis=1).transpose(2, 1, 0)
        return signals_mat_3d, artifacts_mat_3d
