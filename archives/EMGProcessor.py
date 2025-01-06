import numpy as np
from scipy.signal import iirnotch, filtfilt
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
class EMGProcessor:
    def __init__(self, time, EMG_signal, triggers, triggers_time_stamps, fs):
        """
        Initialize the EMGProcessor with EMG data and parameters.

        Parameters:
        EMG_signal (numpy array): The multi-channel EMG signal.
        triggers (numpy array): Array of trigger codes.
        triggers_time_stamps (numpy array): Time stamps corresponding to each trigger.
        fs (float): Sampling frequency of the EMG signal.
        """
        self.EMG_signal = EMG_signal
        self.triggers = triggers
        self.triggers_time_stamps = triggers_time_stamps
        self.fs = fs
        self.time = time
        self.EMG_chunks = []


    def extract_aligned_chunks(self, data, start_stop,start_stop_stim,stim_mode,stage):
        start_indices = np.where(self.triggers == start_stop[0])[0]
        end_indices = np.where(self.triggers == start_stop[1])[0]
        s_start_idxs = np.where(self.triggers == start_stop_stim[0])[0]
        s_stop_idxs = np.where(self.triggers == start_stop_stim[1])[0]
        print()

        if stage == 'fixation':
            stim_start_indices = s_start_idxs[::2]
            stim_stop_indices = s_stop_idxs[::2]
        else:
            stim_start_indices = s_start_idxs[1::2]
            stim_stop_indices = s_stop_idxs[1::2]



        aligned_emg_chunks = []

        time_history = 0
        for idx, (start_idx, end_idx) in enumerate(zip(start_indices, end_indices)):

            start_time = self.triggers_time_stamps[start_idx]
            end_time = self.triggers_time_stamps[end_idx]


            emg_start_idx = (np.abs(self.time - start_time)).argmin()
            emg_end_idx = (np.abs(self.time - end_time)).argmin()

            stim_start_time = self.triggers_time_stamps[stim_start_indices[idx]]
            stim_end_time = self.triggers_time_stamps[stim_stop_indices[idx]]

            emg_stim_start_idx = (np.abs(self.time - stim_start_time)).argmin()
            emg_stim_end_idx = (np.abs(self.time - stim_end_time)).argmin()


            if emg_end_idx <= emg_start_idx:
                print(f"Invalid chunk: start_idx ({emg_start_idx}) >= end_idx ({emg_end_idx})")
                continue
            info = {}

            offset = self.time[emg_start_idx]
            info['emg_chunk'] = self.EMG_signal[emg_start_idx:emg_end_idx, :]
            info['time_chunk'] = self.time[emg_start_idx:emg_end_idx]
            info['stim_range'] = (self.time[emg_stim_start_idx], self.time[emg_stim_end_idx])

            # padding = np.zeros((300,16))  # Adjust the shape and data type as needed
            # padded_trial = np.concatenate((padding, info['emg_chunk']),axis = 0)
            # offset = stim_start_indices[idx] - emg_start_idx
            # shifted_chunk = np.roll(padded_trial,-offset,axis = 0)
            aligned_emg_chunks.append(info)
            time_history = time_history + end_time



        self.EMG_chunks = aligned_emg_chunks

        # # Plotting
        # for chunk in aligned_emg_chunks:
        #     plt.plot(chunk['emg_chunk'][:,1])  # Plot each EMG chunk
        #     plt.axvline(x=chunk['stim_range'][0], color='r', linestyle='--')  # Start of stimulation
        #     plt.axvline(x=chunk['stim_range'][1], color='b', linestyle='--')  # End of stimulation
        #
        # plt.xlabel("Time")
        # plt.ylabel("EMG Signal")
        # plt.title("Aligned EMG Chunks with Stimulation Start and End")
        # plt.show()

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
            self.EMG_signal = filtfilt(b, a, self.EMG_signal, axis=0)

    def dft_multichannel(self, nfft, axis=-1):
        f = np.fft.fftfreq(nfft, 1 / self.fs)
        f_keep = f > 0
        f = f[f_keep]

        n_channels = self.EMG_signal.shape[-1]
        fft_values = np.zeros((n_channels, len(f)))

        for i in range(n_channels):
            y = np.abs(np.fft.fft(self.EMG_signal[:, i], nfft, axis=axis))[f_keep]
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
        self.EMG_signal = filtfilt(b, a, self.EMG_signal, axis=0)

# Example usage:
# emg_processor = EMGProcessor(EMG_signal, triggers, triggers_time_stamps, fs)
# aligned_chunks = emg_processor.extract_aligned_chunks(data, start_stop, stim_trigger)
# filtered_signal = emg_processor.apply_notch_filters([50, 100])
# freq, fft_values = emg_processor.dft_multichannel(nfft)
