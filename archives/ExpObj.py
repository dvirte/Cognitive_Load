import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
from scipy.signal import iirnotch, filtfilt
from scipy.signal import butter, filtfilt
from scipy import signal
from scipy.fft import fftshift
import pandas as pd


class ExpObj():
    def __init__(self, emg_data, trigger_stream, fs=250):
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
        self.fs = fs
        self.eeg_data_alpha = self.apply_bandpass_filter(8, 13)
        self.eeg_data_beta = self.apply_bandpass_filter(13, 30)
        self.eeg_data_theta = self.apply_bandpass_filter(4, 7)
        self.eeg_data_delta = self.apply_bandpass_filter(1, 4)
        self.emg_complementary_data = self.apply_bandpass_filter(1, 35)
        self.apply_notch_filters([50, 100])
        if fs == 250:
            self.emg_data = self.apply_bandpass_filter(35, 124)
        else:
            self.emg_data = self.apply_bandpass_filter(35, 245)
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

    def plot_multiple_channels(self, trail_emg, trail_eeg, bands_names, channel_list=None):
        """
        Plot the EMG and EEG data of the given channels

        :param trail_emg: EMG data of the trail
        :param trail_eeg: EEG data of the trail
        :param bands_names: Names of the bands
        :param channel_list: List of channels to plot
        """

        if channel_list is not None:
            for i in channel_list:
                fig, axs = plt.subplots(2, 1, figsize=(15, 10))

                # Plot the EMG data
                axs[0].plot(trail_emg[1], trail_emg[0][:, i])
                axs[0].set_title(bands_names[0])
                axs[0].set_xlabel('Time (s)')
                axs[0].set_ylabel('Amplitude')
                axs[0].grid(True)
                axs[0].set_title(bands_names[0])

                # Plot the EEG data
                axs[1].plot(trail_eeg[1], trail_eeg[0][:, i])
                axs[1].set_title(bands_names[1])
                axs[1].set_xlabel('Time (s)')
                axs[1].set_ylabel('Amplitude')
                axs[1].grid(True)
                axs[1].set_title(bands_names[1])

                fig.suptitle('Channel ' + str(i))
                plt.show(block=False)

                # Plot spectrogram of the EMG and EEG data
                plt.figure(figsize=(10, 6))
                f, t, Sxx = signal.spectrogram(trail_emg[0][:, i], self.fs)
                plt.pcolormesh(t, f, Sxx, shading='gouraud')
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [sec]')
                plt.title('EMG Spectrogram - Channel ' + str(i))
                plt.show(block=False)

                plt.figure(figsize=(10, 6))
                f, t, Sxx = signal.spectrogram(trail_eeg[0][:, i], self.fs)
                plt.pcolormesh(t, f, Sxx, shading='gouraud')
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [sec]')
                plt.title('EEG Spectrogram - Channel ' + str(i))

            plt.show(block=False)

        else:
            print('Please provide a list of channels to plot')