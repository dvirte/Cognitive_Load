import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
from scipy.signal import iirnotch, filtfilt
from scipy.signal import butter, filtfilt
import pandas as pd


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
        self.eeg_data_alpha = self.apply_bandpass_filter(8, 13)
        self.eeg_data_beta = self.apply_bandpass_filter(13, 30)
        self.eeg_data_theta = self.apply_bandpass_filter(4, 7)
        self.eeg_data_delta = self.apply_bandpass_filter(1, 4)
        self.emg_complementary_data = self.apply_bandpass_filter(1, 35)
        self.apply_notch_filters([50, 100])
        self.emg_data = self.apply_bandpass_filter(35, 124)
        self.list_of_elements = ['EEG_alpha', 'EEG_beta', 'EEG_theta', 'EEG_delta', 'EMG', 'EMG_complementary']
        self.relevant_data, self.relevant_name = self.extract_relevant_electrodes()

    def find_periods(self):
        """
        Identifies the rest periods based on triggers and their corresponding time stamps.
        """
        rest_periods = []
        play_periods = []
        calibrate_periods = []
        cal_ind = None
        start_rest = None

        for i, trigger in enumerate(self.triggers):
            if trigger == 4:
                start_play = i
            elif trigger == 9:
                play_periods.append((self.triggers_time_stamps[start_play], self.triggers_time_stamps[i]))
            elif trigger == 11:
                start_rest = i
            elif trigger == 8 and start_rest is not None:
                rest_periods.append((self.triggers_time_stamps[start_rest], self.triggers_time_stamps[i]))
            elif 13 <= trigger <= 18:
                if cal_ind is not None:
                    calibrate_periods.append((self.triggers_time_stamps[cal_ind], self.triggers_time_stamps[i]))
                cal_ind = i

        while True:
            if self.triggers[i - 1] == 9:
                break
            elif self.triggers[i - 1] == 4:
                # pop the last element in the rest_periods list
                last_start, last_end = rest_periods.pop()
                rest_periods.append((last_start, self.triggers_time_stamps[-1]))
                break
            else:
                i -= 1

        return rest_periods, play_periods, calibrate_periods

    def extract_band_trials(self, status, period, kind='EMG'):
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

        # Choose the element to extract
        if kind == 'EEG_alpha':
            data = self.eeg_data_alpha
        elif kind == 'EEG_beta':
            data = self.eeg_data_beta
        elif kind == 'EEG_theta':
            data = self.eeg_data_theta
        elif kind == 'EEG_delta':
            data = self.eeg_data_delta
        elif kind == 'EMG':
            data = self.emg_data
        elif kind == 'EMG_complementary':
            data = self.emg_complementary_data
        elif kind == 'relevant':
            data = self.relevant_data
        else:
            raise ValueError("kind must be 'EEG_alpha', 'EEG_beta', 'EEG_theta', 'EEG_delta', 'EMG' or 'relevant'")

        # Find the indices of the time stamps that fall within the game time stamps
        start_idx = np.searchsorted(self.data_timestamps, start)
        if self.data_timestamps[start_idx] > start:
            start_idx = max(start_idx - 1, 0)
        stop_idx = np.searchsorted(self.data_timestamps, stop, side='right')
        if self.data_timestamps[stop_idx] < stop:
            start_idx = min(stop_idx + 1, len(self.data_timestamps))
        mask = ((self.triggers_time_stamps >= self.data_timestamps[start_idx]) &
                (self.triggers_time_stamps <= self.data_timestamps[stop_idx]))
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

        return (data[start_idx:stop_idx, :], self.data_timestamps[start_idx:stop_idx],
                self.triggers[indices], self.triggers_time_stamps[indices], (status, period, kind))

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

    def extract_relevant_electrodes(self):
        # Extract electrodes 14-16 for EEG bands and EMG complementary
        eeg_alpha_relevant = self.eeg_data_alpha[:, 0:4]
        eeg_beta_relevant = self.eeg_data_beta[:, 0:4]
        eeg_theta_relevant = self.eeg_data_theta[:, 0:4]
        eeg_delta_relevant = self.eeg_data_delta[:, 0:4]
        emg_complementary_relevant = self.emg_complementary_data[:, 0:4]

        # Keep all electrodes for EMG
        emg_relevant = self.emg_data

        # Combine all relevant electrodes into a single set
        combined_data = np.hstack((
            eeg_alpha_relevant,
            eeg_beta_relevant,
            eeg_theta_relevant,
            eeg_delta_relevant,
            emg_complementary_relevant,
            emg_relevant
        ))

        # List of relevant elements and channels
        relevant_elements = ['EEG_alpha_ch16', 'EEG_alpha_ch15', 'EEG_alpha_ch14', 'EEG_alpha_ch13', 'EEG_beta_ch16',
                             'EEG_beta_ch15', 'EEG_beta_ch14', 'EEG_beta_ch13', 'EEG_theta_ch16', 'EEG_theta_ch15',
                             'EEG_theta_ch14', 'EEG_theta_ch13', 'EEG_delta_ch16', 'EEG_delta_ch15', 'EEG_delta_ch14',
                             'EEG_delta_ch13', 'EMG_complementary_ch16', 'EMG_complementary_ch15',
                             'EMG_complementary_ch14', 'EMG_complementary_ch13', 'EMG_ch16', 'EMG_ch15', 'EMG_ch14',
                             'EMG_ch13', 'EMG_ch12', 'EMG_ch11', 'EMG_ch10', 'EMG_ch9', 'EMG_ch8', 'EMG_ch7',
                             'EMG_ch6', 'EMG_ch5', 'EMG_ch4', 'EMG_ch3', 'EMG_ch2', 'EMG_ch1']

        return combined_data, relevant_elements

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
        data, timestamps, triggers, trigger_times = trail[0], trail[1], trail[2], trail[3]
        trigger_times = trigger_times - timestamps[0]
        timestamps = timestamps - timestamps[0]  # Convert to seconds
        status, period, kind = trail[4]
        samples, channels = data.shape

        # Define the title based on the status and period
        if status == 0:
            title = f'{kind} - Rest Period: {period}'
        elif status == 1:
            title = f'{kind} - Task Period: {period}'
        elif status == 2:
            title = f'{kind} - Calibration'

        # Determine the global min and max values across all channels
        global_min = np.min(data)
        global_max = np.max(data)

        fig, axs = plt.subplots(channels, 1, figsize=(10, 20), sharex=True)  # 16 rows, 1 col, shared X-axis
        fig.suptitle(title, fontsize=16)  # Add title to the plot

        for i in range(channels):
            axs[i].plot(timestamps, data[:, channels - 1 - i])  # Plot each channel
            axs[i].text(1.02, 0.5, f'EMG {channels - i}', transform=axs[i].transAxes, va='center', ha='left')
            axs[i].spines['top'].set_visible(False)  # Hide the top spine
            axs[i].spines['right'].set_visible(False)  # Hide the right spine
            axs[i].spines['left'].set_visible(False)  # Optionally, hide the left spine as well
            axs[i].spines['bottom'].set_visible(False)  # Hide the bottom spine
            axs[i].get_xaxis().set_visible(False)  # Hide x-axis labels and ticks
            axs[i].set_ylim(global_min, global_max)  # Set the same Y-axis limits for all subplots

            # Add vertical red lines for triggers
            for trigger_time in trigger_times:
                axs[i].axvline(x=trigger_time, color='red', linestyle='--')

        # Add trigger labels on the last subplot
        for j, trigger_time in enumerate(trigger_times):
            axs[-1].text(trigger_time, axs[-1].get_ylim()[0] - 0.05 * (axs[-1].get_ylim()[1] - axs[-1].get_ylim()[0]),
                         f'Trigger {int(triggers[j])}', color='red', ha='center', va='top', rotation=90)

        # Adjust settings for the last subplot
        axs[-1].spines['bottom'].set_visible(True)  # Show the bottom spine for the last subplot
        axs[-1].get_xaxis().set_visible(True)  # Show x-axis labels and ticks for the last subplot
        axs[-1].set_xlabel("Time [s]")  # Common X-axis title
        fig.text(0.04, 0.5, 'Amplitude [mV]', va='center', rotation='vertical')  # Common Y-axis title

        plt.subplots_adjust(hspace=0)  # Remove horizontal space between plots
        plt.show(block=False)
        input("Press Enter to keep going...")

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
        return filtfilt(b, a, self.data, axis=0)

    def calculate_stat(self, data, window_size, overlap, stat):
        """
        Calculate the statistical values for each window.

        Parameters:
        data (ndarray): The data to calculate the RMS for.
        window_size (float): The size of the window in seconds.
        overlap (float): The overlap between windows as a fraction.

        Returns:
        ndarray: The statistical values for each window.
        """
        window_size = int(window_size * self.fs)
        overlap = int(overlap * window_size)
        stat_values = []

        if stat == 'rms':
            for i in range(0, data[0].shape[0] - window_size, window_size - overlap):
                window = data[0][i:i + window_size, :]
                stat_values.append(np.sqrt(np.mean(window ** 2, axis=0)))
        elif stat == 'energy':
            for i in range(0, data[0].shape[0] - window_size, window_size - overlap):
                window = data[0][i:i + window_size, :]
                stat_values.append(np.sum(window ** 2, axis=0))

        return np.array(stat_values)

    def stat_band(self, status, period, window_size=0.5, overlap=0.5, stat='rms', relevant=True):
        """
        Calculate the statistical values for each frequency band.
        Parameters:
        status (int): 0 for rest, 1 for task, 2 for calibration.
        period (int): the specific period to extract the data from.
        window_size (float): The size of the window in seconds.
        overlap (float): The overlap between windows as a fraction.
        :return:
        stat_alpha: statistical values for the alpha band
        stat_beta: statistical values for the beta band
        stat_theta: statistical values for the theta band
        stat_delta: statistical values for the delta band
        stat_emg: statistical values for the EMG band
        stat_emg_complementary: statistical values for the EMG complementary band
        """
        if relevant:
            relevant_data = self.extract_band_trials(status, period, 'relevant')
            return self.calculate_stat(relevant_data, window_size, overlap, stat)
        else:
            # Extract the EEG data for the specific level and response status
            eeg_alpha = self.extract_band_trials(status, period, 'EEG_alpha')
            eeg_beta = self.extract_band_trials(status, period, 'EEG_beta')
            eeg_theta = self.extract_band_trials(status, period, 'EEG_theta')
            eeg_delta = self.extract_band_trials(status, period, 'EEG_delta')
            emg = self.extract_band_trials(status, period, 'EMG')
            emg_complementary = self.extract_band_trials(status, period, 'EMG_complementary')

            # Calculate the RMS for each frequency band by windowing the data
            stat_alpha = self.calculate_stat(eeg_alpha, window_size, overlap, stat)
            stat_beta = self.calculate_stat(eeg_beta, window_size, overlap, stat)
            stat_theta = self.calculate_stat(eeg_theta, window_size, overlap, stat)
            stat_delta = self.calculate_stat(eeg_delta, window_size, overlap, stat)
            stat_emg = self.calculate_stat(emg, window_size, overlap, stat)
            stat_emg_complementary = self.calculate_stat(emg_complementary, window_size, overlap, stat)

            return stat_alpha, stat_beta, stat_theta, stat_delta, stat_emg, stat_emg_complementary

    def plot_stat(self, stat_values, status, period, kind='RMS', specific_bands=None):
        """
        Plot the statistical values for each frequency band.

        Parameters:
        stat_values (tuple): The statistical values for each frequency band.
        status (int): 0 for rest, 1 for task, 2 for calibration.
        period (int): the specific period to extract the data from.

        Returns:
        None: The statistical values are plotted.
        """
        stat_alpha, stat_beta, stat_theta, stat_delta, stat_emg, stat_emg_complementary = stat_values

        # Create a time axis for the RMS values
        time = np.arange(0, len(stat_alpha)) * 0.25

        # Define the title based on the status and period
        channels = 16

        all_frequency_bands = {
            'Alpha': stat_alpha,
            'Beta': stat_beta,
            'Theta': stat_theta,
            'Delta': stat_delta,
            'EMG': stat_emg,
            'EMG_complementary': stat_emg_complementary
        }

        if specific_bands is None:
            frequency_bands = all_frequency_bands
        else:
            frequency_bands = {band: all_frequency_bands[band] for band in specific_bands if
                               band in all_frequency_bands}

        for band, data in frequency_bands.items():
            title = f'{kind} Values for {band} Band - Status: {status}, Period: {period}'
            fig, axs = plt.subplots(channels, 1, figsize=(10, 20), sharex=True)  # 16 rows, 1 col, shared X-axis
            fig.suptitle(title, fontsize=16)  # Add title to the plot

            # Determine the global min and max values across all channels
            global_min = np.min(data)
            global_max = np.max(data)

            for i in range(channels):
                axs[i].plot(time, data[:, channels - 1 - i])
                axs[i].text(1.02, 0.5, f'Channel {channels - i}', transform=axs[i].transAxes, va='center', ha='left')
                axs[i].spines['top'].set_visible(False)  # Hide the top spine
                axs[i].spines['right'].set_visible(False)  # Hide the right spine
                axs[i].spines['left'].set_visible(False)  # Optionally, hide the left spine as well
                axs[i].spines['bottom'].set_visible(False)  # Hide the bottom spine
                axs[i].get_xaxis().set_visible(False)  # Hide x-axis labels and ticks
                axs[i].set_ylim(global_min, global_max)  # Set the same Y-axis limits for all subplots

            # Adjust settings for the last subplot
            axs[-1].spines['bottom'].set_visible(True)  # Show the bottom spine for the last subplot
            axs[-1].get_xaxis().set_visible(True)  # Show x-axis labels and ticks for the last subplot
            axs[-1].set_xlabel("Time [s]")  # Common X-axis title
            fig.text(0.04, 0.5, 'RMS', va='center', rotation='vertical')  # Common Y-axis title

            plt.subplots_adjust(hspace=0)  # Remove horizontal space between plots
            plt.show(block=False)
        input("Press Enter to keep going...")

    def extract_statistics(self, trail):
        """
        Extract statistical values for each filter in the trail.

        Parameters:
        trail (tuple of ndarrays): A tuple containing 6 different bands, each an ndarray with 16 columns (channels).
        list of bands: ['Alpha', 'Beta', 'Theta', 'Delta', 'emg', 'complementary emg']

        Returns:
        stats_df (DataFrame): A DataFrame containing the statistical values for each channel and filter.
        """
        stats = ['mean', 'std', 'median', 'min', 'max', 'range', 'variance', 'iqr']
        data = []

        if type(trail) is tuple:
            for band_idx, filter_data in enumerate(trail):
                for channel_idx in range(filter_data.shape[1]):
                    channel_data = filter_data[:, channel_idx]
                    mean = np.mean(channel_data)
                    std = np.std(channel_data)
                    median = np.median(channel_data)
                    min_val = np.min(channel_data)
                    max_val = np.max(channel_data)
                    range_val = max_val - min_val
                    variance = np.var(channel_data)
                    iqr = np.percentile(channel_data, 75) - np.percentile(channel_data, 25)

                    data.append([band_idx, channel_idx, mean, std, median, min_val, max_val, range_val, variance, iqr])

                stats_df = pd.DataFrame(data, columns=['Filter', 'Channel', 'Mean', 'STD', 'Median', 'Min', 'Max',
                                                       'Range', 'Variance', 'IQR'])
        else:
            for channel_idx in range(np.shape(trail)[1]):
                channel_data = trail[:, channel_idx]
                mean = np.mean(channel_data)
                std = np.std(channel_data)
                median = np.median(channel_data)
                min_val = np.min(channel_data)
                max_val = np.max(channel_data)
                range_val = max_val - min_val
                variance = np.var(channel_data)
                iqr = np.percentile(channel_data, 75) - np.percentile(channel_data, 25)

                data.append([self.relevant_name[channel_idx], mean, std, median, min_val, max_val, range_val, variance, iqr])

            stats_df = pd.DataFrame(data, columns=['name', 'Mean', 'STD', 'Median', 'Min', 'Max', 'Range',
                                                   'Variance', 'IQR'])

        return stats_df

    def plot_rms_multiple(self, rms_values_list, status_list, period_list):
        """
        Plot the RMS values for each frequency band for multiple trails.

        Parameters:
        rms_values_list (list of tuples): A list containing tuples of RMS values for each frequency band for different trails.
        status_list (list of int): A list containing status values for each trail.
        period_list (list of int): A list containing period values for each trail.

        Returns:
        None: The RMS values are plotted.
        """
        # Define the frequency bands and their corresponding data from the first trail
        frequency_bands = ['Alpha', 'Beta', 'Theta', 'Delta', 'emg', 'complementary emg']

        channels = 16

        for band_idx, band in enumerate(frequency_bands):
            title = f'RMS Values for {band} Band - Multiple Trails'
            fig, axs = plt.subplots(channels, 1, figsize=(10, 20), sharex=True)  # 16 rows, 1 col, shared X-axis
            fig.suptitle(title, fontsize=16)  # Add title to the plot

            for i in range(channels):
                for j, (rms_values, status, period) in enumerate(zip(rms_values_list, status_list, period_list)):
                    # Create a time axis for the RMS values (assuming all trails have the same length)
                    time = np.arange(0, len(rms_values_list[j][0])) * 0.5
                    axs[i].plot(time, rms_values[band_idx][:, i],
                                label=f'Trail {j + 1} - Status: {status}, Period: {period}')
                axs[i].text(1.02, 0.5, f'Channel {i + 1}', transform=axs[i].transAxes, va='center', ha='left')
                axs[i].spines['top'].set_visible(False)  # Hide the top spine
                axs[i].spines['right'].set_visible(False)  # Hide the right spine
                axs[i].spines['left'].set_visible(False)  # Optionally, hide the left spine as well
                axs[i].spines['bottom'].set_visible(False)  # Hide the bottom spine
                axs[i].get_xaxis().set_visible(False)  # Hide x-axis labels and ticks

            # Adjust settings for the last subplot
            axs[-1].spines['bottom'].set_visible(True)  # Show the bottom spine for the last subplot
            axs[-1].get_xaxis().set_visible(True)  # Show x-axis labels and ticks for the last subplot
            axs[-1].set_xlabel("Time [s]")  # Common X-axis title
            fig.text(0.04, 0.5, 'RMS', va='center', rotation='vertical')  # Common Y-axis title

            # Create a common legend for the figure
            handles, labels = axs[0].get_legend_handles_labels()
            plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 17.5), ncol=len(rms_values_list))

            plt.subplots_adjust(hspace=0)  # Remove horizontal space between plots
            plt.show(block=False)

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