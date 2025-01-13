import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import emd
from collections import Counter
import copy


def plot_emd_signal(data_obj, maze_index):

    # get data from the first 3 blinks - 0.5 seconds before and 0.5 seconds after
    # Isolate the EMG signal for the given maze
    blink_times = data_obj.calibrate_periods[0]
    blink_start = blink_times[0]
    blink_end = blink_times[-1]
    blink_segment, time_segment = segment_data(data_obj, blink_start, blink_end)

    # Plot the EMG signal for the specified maze
    plot_signal(blink_segment, time_segment, maze_index, calibration = True)


    # Isolate the EMG signal for the given maze
    start_time, end_time = data_obj.play_periods[maze_index]
    print(start_time, end_time)
    emg_segment, time_segment = segment_data(data_obj, start_time, end_time)
    full_period = copy.deepcopy(emg_segment)

    # Plot the EMG signal for the specified maze
    plot_signal(emg_segment, time_segment, maze_index)

    # Run EMD on the EMG signal only on for channels 14-16
    emg_segment = emg_segment[:, 13:16]
    std_list = []
    count_index = []
    for i in range(3):
        imf = emd.sift.sift(emg_segment[:,i])
        std_list.append(calc_std_peaks(imf, 2))
        emd.plotting.plot_imfs(imf)
        count_index += std_list[i][1][0].tolist()
        title = f"EMD IMFs - Channel {i + 14}"
        plt.suptitle(title, fontsize=16)

    number_counts = Counter(count_index)
    # Keep numbers that appear in at least two lists
    result = [num for num, count in number_counts.items() if count >= 2]

    # Plot the standard deviation of the IMFs
    fig, axs = plt.subplots(len(std_list), 1, figsize=(10, 20), sharex=True)  # 16 rows, 1 col, shared X-axis
    fig.suptitle("Standard Deviation of IMFs", fontsize=16)  # Add title to the plot
    for i in range(len(std_list)):
        axs[i].plot(std_list[i][0])
        axs[i].scatter(std_list[i][1][0], std_list[i][1][1]['peak_heights'], color='r')
    plt.show(block=False)

    # Plot the real peaks
    real_peaks = calc_real_peaks(full_period, result)
    plot_signal(full_period, time_segment, maze_index, peaks=real_peaks)

def segment_data(data_obj, start_time, end_time):
    start_idx = np.searchsorted(data_obj.data_timestamps, start_time)
    end_idx = np.searchsorted(data_obj.data_timestamps, end_time)
    return data_obj.data[start_idx:end_idx], data_obj.data_timestamps[start_idx:end_idx]

def calc_std_peaks(imf, channel_imf, window = 0.3, fs=250, threshold  =0.5):
    signal_imf = imf[:, channel_imf]
    num_of_samples = window * fs
    std_list = []
    for i in range(0, len(signal_imf), int(num_of_samples)):
        std_list.append(np.std(signal_imf[i:i+int(num_of_samples)]))
    # Find the peaks
    std_peaks = signal.find_peaks(std_list, height=threshold*max(std_list))
    return std_list, std_peaks

def calc_real_peaks(emg_segment, std_ind, window = 0.3, fs=250):
    # convert the index of the std to the range of the window
    num_of_samples = int(window * fs)
    real_peaks = []
    for i in std_ind:
        real_peaks.append(list(range(i * num_of_samples, i * num_of_samples + num_of_samples)))

    max_indexes = []
    for i in range(len(real_peaks)):
        max_indexes.append(np.argmax(emg_segment[real_peaks[i], 15])+real_peaks[i][0])
    return max_indexes

def plot_signal(emg_segment, time_segment, maze_index, calibration = False, peaks = None):

    # Plot original EMG signal
    global_min = np.min(emg_segment)
    global_max = np.max(emg_segment)
    channels = emg_segment.shape[1]
    time_segment -= time_segment[0]  # Set the start time to 0
    if calibration:
        title = f"EMG Signal - Calibration"
    else:
        if peaks is not None:
            title = f"EMG Signal - Maze {maze_index + 1} with peaks"
        else:
            title = f"EMG Signal - Maze {maze_index + 1}"

    fig, axs = plt.subplots(channels, 1, figsize=(10, 20), sharex=True)  # 16 rows, 1 col, shared X-axis
    fig.suptitle(title, fontsize=16)  # Add title to the plot

    for i in range(channels):
        axs[i].plot(time_segment, emg_segment[:, channels - 1 - i])  # Plot each channel
        axs[i].text(1.02, 0.5, f'EMG {channels - i}', transform=axs[i].transAxes, va='center', ha='left')
        axs[i].spines['top'].set_visible(False)  # Hide the top spine
        axs[i].spines['right'].set_visible(False)  # Hide the right spine
        axs[i].spines['left'].set_visible(False)  # Optionally, hide the left spine as well
        axs[i].spines['bottom'].set_visible(False)  # Hide the bottom spine
        axs[i].get_xaxis().set_visible(False)  # Hide x-axis labels and ticks
        axs[i].set_ylim(global_min, global_max)  # Set the same Y-axis limits for all subplots

        if peaks is not None and i < 5:
            axs[i].scatter(time_segment[peaks], emg_segment[peaks, channels - 1 - i], color='r')

    # Adjust settings for the last subplot
    axs[-1].spines['bottom'].set_visible(True)  # Show the bottom spine for the last subplot
    axs[-1].get_xaxis().set_visible(True)  # Show x-axis labels and ticks for the last subplot
    axs[-1].set_xlabel("Time [s]")  # Common X-axis title
    fig.text(0.04, 0.5, 'Amplitude [mV]', va='center', rotation='vertical')  # Common Y-axis title

    plt.subplots_adjust(hspace=0)  # Remove horizontal space between plots
    plt.show(block=False)