import numpy as np
import pywt
import copy
import matplotlib.pyplot as plt

from docutils.nodes import title


def extract_features(data_obj, maze_index, window_size = 0.3, wavelet='db7'):
    """
    Extract features from the EMG signal for a given maze.

    :param data_obj:
    :param maze_index:
    :param wavelet:

    :return:
    channel_features: A list of features for each channel.

    """
    # Get the start and end time of the maze
    start_time, end_time = data_obj.play_periods[maze_index]

    # Segment the data and timestamps for the maze
    emg_segment, time_segment = segment_data(data_obj, start_time, end_time)
    emg_segment_relevant = emg_segment[:, 0:3]  # Only use channels 14-16

    # Loop through the EMG signal in 300 ms windows
    num_of_samples = int(data_obj.fs * window_size)


    # Prepare to store features for each channel
    list_of_features = [[], [], [], [], []] # MAV_CD4, STD_CD4, MAV_CD5, STD_CD5, MAV_x
    channel_features = []
    for i in range(np.shape(emg_segment_relevant)[1]):
        channel_features.append(copy.deepcopy(list_of_features))

    # Loop through the EMG signal in 300 ms windows
    for i in range(0, len(emg_segment_relevant), num_of_samples):
        emg_window = emg_segment_relevant[i:i+num_of_samples]

        # Calculate CD4 and CD5 coefficients
        cd4, cd5 = calculate_cd(emg_window, wavelet)

        # Calculate features from CD4, CD5, and the raw signal
        features = calculate_features(cd4, cd5, emg_window)

        # Append to the list of features
        for j in range(len(features)):
            for channel in range(np.shape(emg_segment_relevant)[1]):
                val = features[j][channel]
                (channel_features[channel][j].append(val))

    # Plot the statistics
    plot_statistic(channel_features, window_size)

    return channel_features

def calculate_cd(emg_segment, wavelet='db7'):
    """
    Calculate CD4 and CD5 coefficients for a given maze.

    Parameters:
    - emg_segment: The EMG signal for the 300 ms window (array-like).
    - wavelet: The wavelet to use for the Discrete Wavelet Transform.

    Returns:
    - cd4: A list of CD4 coefficients for each channel.
    - cd5: A list of CD5 coefficients for each channel.
    """
    # Prepare to store CD4 and CD5 for each channel
    cd4_all_channels = []
    cd5_all_channels = []

    # Loop through each channel and calculate CD4 and CD5
    for channel_idx in range(emg_segment.shape[1]):  # Iterate over 16 channels
        channel_data = emg_segment[:, channel_idx]

        # Perform Discrete Wavelet Transform (DWT)
        coeffs = pywt.wavedec(channel_data, wavelet, level=5)

        # Extract CD4 and CD5 correctly
        cd5 = coeffs[1]  # CD5 is at level 5
        cd4 = coeffs[2]  # CD4 is at level 4

        # Append to results
        cd4_all_channels.append(cd4)
        cd5_all_channels.append(cd5)

    return cd4_all_channels, cd5_all_channels

def segment_data(data_obj, start_time, end_time):
    start_idx = np.searchsorted(data_obj.data_timestamps, start_time)
    end_idx = np.searchsorted(data_obj.data_timestamps, end_time)
    return data_obj.data[start_idx:end_idx], data_obj.data_timestamps[start_idx:end_idx]

def calculate_features(cd4, cd5, raw_signal):
    """
    Calculate features (MAV, STD) from CD4, CD5, and the raw EMG signal.

    Parameters:
    - cd4: Coefficients from CD4 (array-like).
    - cd5: Coefficients from CD5 (array-like).
    - raw_signal: The original raw EMG signal for the 300 ms window (array-like).

    Returns:
    - mav_cd4_list: A list of MAV values for CD4 for each channel.
    - std_cd4_list: A list of STD values for CD4 for each channel.
    - mav_cd5_list: A list of MAV values for CD5 for each channel.
    - std_cd5_list: A list of STD values for CD5 for each channel.
    - mav_raw_list: A list of MAV values for the raw signal for each channel.
    """
    # Calculate MAV and STD for CD4
    mav_cd4_list = []
    std_cd4_list = []
    mav_cd5_list = []
    std_cd5_list = []
    mav_raw_list = []

    # Iterate through each channel's coefficients
    for i in range(len(cd4)):
        # Extract coefficients for the current channel
        channel_cd4 = cd4[i]
        channel_cd5 = cd5[i]
        channel_raw_signal = raw_signal[:, i]

        # Calculate MAV and STD for CD4
        mav_cd4 = np.mean(np.abs(channel_cd4))  # Mean Absolute Value for CD4
        std_cd4 = np.std(channel_cd4)  # Standard Deviation for CD4

        # Calculate MAV and STD for CD5
        mav_cd5 = np.mean(np.abs(channel_cd5))  # Mean Absolute Value for CD5
        std_cd5 = np.std(channel_cd5)  # Standard Deviation for CD5

        # Calculate MAV for the raw signal (time-domain signal)
        mav_raw = np.mean(np.abs(channel_raw_signal))  # Mean Absolute Value for the raw signal

        # Append to lists
        mav_cd4_list.append(mav_cd4)
        std_cd4_list.append(std_cd4)
        mav_cd5_list.append(mav_cd5)
        std_cd5_list.append(std_cd5)
        mav_raw_list.append(mav_raw)

    return mav_cd4_list, std_cd4_list, mav_cd5_list, std_cd5_list, mav_raw_list

def plot_statistic(features, window_size=0.3):

    list_of_stat = ['MAV_CD4', 'STD_CD4', 'MAV_CD5', 'STD_CD5', 'MAV_x']

    for channel in range(len(features)):
        fig, axs = plt.subplots(5, 1, figsize=(10, 20), sharex=True)

        title = f"Channel {channel + 1} - Statistics"
        fig.suptitle(title, fontsize=16)


        # Plot original EMG signal
        global_min = np.min(features[channel])
        global_max = np.max(features[channel])
        num_stat = len(features[1])

        time_segment = np.arange(0, len(features[channel][0])) * window_size

        for i in range(num_stat):
            axs[i].plot(time_segment, features[channel][i])  # Plot each channel
            axs[i].text(1.02, 0.5, list_of_stat[i], transform=axs[i].transAxes, va='center', ha='left')
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
        fig.text(0.04, 0.5, 'statistic', va='center', rotation='vertical')  # Common Y-axis title


        plt.subplots_adjust(hspace=0)  # Remove horizontal space between plots
        plt.show(block=False)