import matplotlib.pyplot as plt
import numpy as np
from astropy.units.quantity_helper.function_helpers import block
from scipy import signal

def extract_blink_info(data_obj, maze_index):
    """
    Extracts blink times for a given maze index from a DataObj.

    :param data_obj: Object containing experimental data (DataObj instance).
    :param maze_index: Integer indicating the maze number (0-based).
    :return: List of blink times (in seconds).
    """
    fs = data_obj.fs  # Sampling frequency
    trigger_stream = data_obj.triggers
    trigger_times = data_obj.triggers_time_stamps

    # Find the start times of blink triggers (13) in calibration phase
    blink_start_indices = np.where(trigger_stream == 13)[0]

    # Get blink times for the calibration phase
    blink_times = trigger_times[blink_start_indices]

    # Isolate the EMG signal for the given maze
    start_time, end_time = data_obj.play_periods[maze_index]
    emg_segment, time_segment = segment_data(data_obj, start_time, end_time)
    emg_segment = emg_segment[:,:6]

    # Filter EMG signal for blink detection
    emg_filtered = preprocess_emg_signal(emg_segment, fs)

    # Create multiple blink templates based on all available blink patterns
    templates = [create_blink_template(data_obj, blink_time) for blink_time in blink_times[:3]]

    # Plot the blink templates for all channels
    for i, template in enumerate(templates):
        # Time vector for the x-axis
        time_vector = np.arange(template.shape[0]) / fs

        # Set up the figure for the 16 channel subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 10))
        fig.suptitle(f"Blink Template {i + 1} - All Channels", fontsize=16)

        # Plot each channel in a subplot
        for channel in range(np.shape(emg_filtered)[1]):
            ax = axes[channel // 2, channel % 2]
            ax.plot(time_vector, template[:, channel])
            ax.set_title(f"Channel {channel + 11}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")

        # Adjust layout and display the plot
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show(block=False)


    # Detect blinks using each template and consolidate the results
    blink_times_detected = []
    for template in templates:
        detected = detect_blinks(emg_filtered, template)
        time_segment_channels = []
        for i in range(len(detected)):
            time_segment_channels.append(time_segment[detected[i]])
        blink_times_detected.append(detected)

    # Plot the EMG signal with detected blinks
    for i, detected in enumerate(blink_times_detected):
        # Correct time vector for the x-axis in seconds
        time_vector = time_segment

        # Set up the figure for the 6-channel subplots
        fig, axes = plt.subplots(6, 1, figsize=(15, 10))
        fig.suptitle(f"Detected Blinks for Template {i + 1} - All Channels", fontsize=16)

        # Plot each channel in a subplot
        for channel in range(np.shape(emg_filtered)[1]):
            ax = axes[channel]
            ax.plot(time_vector, emg_filtered[:, channel], label="EMG Signal")
            ax.set_title(f"Channel {channel + 11}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")

            # Convert blink indices to time in seconds
            blink_times = time_segment[detected[channel]]  # Convert to seconds
            ax.scatter(blink_times, emg_filtered[detected[channel], channel], color='r',
                       label="Detected Blinks")

        # Adjust layout and display the plot
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show(block=False)


    return blink_times_detected

def segment_data(data_obj, start_time, end_time):
    start_idx = np.searchsorted(data_obj.data_timestamps, start_time)
    end_idx = np.searchsorted(data_obj.data_timestamps, end_time)
    return data_obj.data[start_idx:end_idx], data_obj.data_timestamps[start_idx:end_idx]

def preprocess_emg_signal(emg_data, fs):
    # Notch filters and band-pass filter for blink detection
    b_notch, a_notch = signal.iirnotch(50, 30, fs)
    emg_notched = signal.filtfilt(b_notch, a_notch, emg_data, axis=0)
    b_notch, a_notch = signal.iirnotch(100, 30, fs)
    emg_notched = signal.filtfilt(b_notch, a_notch, emg_notched, axis=0)
    sos = signal.butter(4, [1, 30], btype='bp', fs=fs, output='sos')
    emg_filtered = signal.sosfilt(sos, emg_notched, axis=0)
    return emg_filtered

def create_blink_template(data_obj, blink_start_time):
    fs = data_obj.fs
    blink_duration = 0.1  # Approximate blink duration in seconds
    start_idx = np.searchsorted(data_obj.data_timestamps, blink_start_time)
    end_idx = start_idx + int(blink_duration * fs)
    start_idx = max(0, start_idx - int(0.5 * fs))  # Include a 0.3s window before the blink

    template_filtered = preprocess_emg_signal(data_obj.data[start_idx:end_idx, :], fs)

    # Apply FFT and retain only the first 3 Fourier coefficients
    template_fft = np.fft.fft(template_filtered, axis=0)
    template_fft[3:, :] = 0  # Zero out all coefficients beyond the first three
    template_filtered = np.fft.ifft(template_fft, axis=0)
    template_filtered = np.real(template_filtered) / np.sqrt(np.sum(np.square(template_filtered)))

    return template_filtered[:,:6]

def detect_blinks(emg_data, template):
    """Detects blinks using cross-correlation with overlapping windows across all channels."""
    detected_blinks = []

    for channel in range(emg_data.shape[1]):
        # Continuous cross-correlation with the template
        corr = signal.correlate(emg_data[:, channel], template[:, channel], mode='same')
        peaks_indx, _ = signal.find_peaks(corr, height=0.2)  # Adjust height threshold as needed
        detected_blinks.append(peaks_indx)  # Offset peaks by the window start index

    return detected_blinks
