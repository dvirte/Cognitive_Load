import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from debugpy.launcher import channel
from scipy.signal import iirnotch
from scipy.signal import filtfilt, welch, spectrogram
from scipy.fft import fftshift
from tqdm import tqdm
from fooof import FOOOF
from scipy.stats import spearmanr
import pandas as pd
import seaborn as sns



def plot_trail(trail):
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
    status, period = trail[4]
    samples, channels = data.shape

    # Define the title based on the status and period
    if status == 0:
        title = f'Rest Period: {period}'
    elif status == 1:
        title = f'Task Period: {period}'
    elif status == 2:
        title = f'Calibration'

    # Determine the global min and max values across all channels
    # global_min = np.mean(data)-2*np.std(data)
    # global_max = np.mean(data)+2*np.std(data)



    fig, axs = plt.subplots(channels, 1, figsize=(10, 20), sharex=True)  # 16 rows, 1 col, shared X-axis
    fig.suptitle(title, fontsize=16)  # Add title to the plot

    for i in range(channels):
        global_min = np.min(data[:, channels - 1 - i])
        global_max = np.max(data[:, channels - 1 - i])

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


def plot_signal_with_time(exp_processor, a=1, d=5, n=1):
    """
    Plots a specific time window of the signal for all electrodes with triggers marked,
    based on the given parameters.

    Parameters:
    - exp_processor (ExpProcessor): The experiment processor object containing the data and triggers.
    - a (int): 0 for start-based calculation, 1 for end-based calculation.
    - d (int): Duration of the plot in minutes.
    - n (int): Starting point in minutes from the beginning (if a=0) or end (if a=1) of the signal.

    Returns:
    - None: Displays the plot.
    """
    # Convert time to seconds
    plot_duration = d * 60  # Duration of the plot in seconds
    start_offset = n * 60  # Offset from the start or end of the signal in seconds

    # Get signal and trigger timestamps
    data_timestamps = exp_processor.data_timestamps
    data = exp_processor.data
    trigger_timestamps = exp_processor.triggers_time_stamps
    triggers = exp_processor.triggers

    # Determine the start and end times based on `a`
    if a == 0:  # Start-based calculation
        start_time = start_offset
        end_time = start_time + plot_duration
    elif a == 1:  # End-based calculation
        start_time = data_timestamps[-1] - start_offset
        end_time = start_time + plot_duration
    else:
        raise ValueError("Parameter 'a' must be either 0 (start-based) or 1 (end-based).")

    # Handle edge cases: Ensure the window is within signal bounds
    if end_time > data_timestamps[-1]:
        end_time = data_timestamps[-1]
        start_time = max(0, end_time - plot_duration)
        title = f"Signal Plot of the Last {d} min"
    else:
        title = f"Signal Plot from {n} to {n + d} min since the {'start' if a == 0 else 'end'}"

    # Extract the relevant portion of the signal
    start_idx = np.searchsorted(data_timestamps, start_time)
    end_idx = np.searchsorted(data_timestamps, end_time)
    data_segment = data[start_idx:end_idx, :]
    timestamps = data_timestamps[start_idx:end_idx] - start_time

    # Extract the relevant triggers (including those beyond the signal timestamp range)
    trigger_mask = trigger_timestamps >= start_time
    triggers_segment = triggers[trigger_mask]
    trigger_times = trigger_timestamps[trigger_mask] - start_time

    # Create the plot
    channels = data.shape[1]

    fig, axs = plt.subplots(channels, 1, figsize=(10, 20), sharex=True)  # 16 rows, 1 col, shared X-axis
    # Calculate the adjusted duration in case d exceeds n
    actual_start_time = max(0, end_time - plot_duration if a == 1 else start_time)
    actual_end_time = min(data_timestamps[-1], start_time + plot_duration if a == 0 else end_time)
    actual_duration = (actual_end_time - actual_start_time) / 60  # Convert to minutes

    fig.suptitle(title, fontsize=16)

    for i in range(channels):
        global_min = np.min(data_segment[:, channels - 1 - i])
        global_max = np.max(data_segment[:, channels - 1 - i])

        axs[i].plot(timestamps, data_segment[:, channels - 1 - i])  # Plot each channel
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
                     f'Trigger {int(triggers_segment[j])}', color='red', ha='center', va='top', rotation=90)

    # Adjust settings for the last subplot
    axs[-1].spines['bottom'].set_visible(True)  # Show the bottom spine for the last subplot
    axs[-1].get_xaxis().set_visible(True)  # Show x-axis labels and ticks for the last subplot
    axs[-1].set_xlabel("Time [s]")  # Common X-axis title
    fig.text(0.04, 0.5, 'Amplitude [mV]', va='center', rotation='vertical')  # Common Y-axis title

    plt.subplots_adjust(hspace=0)  # Remove horizontal space between plots
    plt.show(block=False)


def plot_stat(stat_values, status, period, kind='RMS', specific_bands=None):
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


def plot_channel_statistics(list_of_trail, trail_cog_nasa):
    # Assuming all DataFrames have the same structure and channels
    channels = list_of_trail[0].index
    name_channels = list_of_trail[0]['name']

    # Number of trials
    num_trials = len(list_of_trail)


    # Trial numbers
    trial_numbers = np.arange(num_trials)

    for channel in channels:
        plt.figure(figsize=(10, 6))

        # Initialize lists to hold values for each statistical measure
        means, stds, medians, mins, maxs, ranges, variances, iqrs = [], [], [], [], [], [], [], []

        for df in list_of_trail:
            # means.append(df.loc[channel, 'Mean'])
            # stds.append(df.loc[channel, 'STD'])
            # medians.append(df.loc[channel, 'Median'])
            # mins.append(df.loc[channel, 'Min'])
            maxs.append(df.loc[channel, 'Max'])
            ranges.append(df.loc[channel, 'Range'])
            # variances.append(df.loc[channel, 'Variance'])
            # iqrs.append(df.loc[channel, 'IQR'])

        # Plotting each statistical measure
        # plt.plot(trial_numbers, means, label='Mean', marker='o')
        # plt.plot(trial_numbers, stds, label='STD', marker='o')
        # plt.plot(trial_numbers, medians, label='Median', marker='o')
        # plt.plot(trial_numbers, mins, label='Min', marker='o')
        plt.plot(trial_numbers, maxs, label='Max', marker='o')
        plt.plot(trial_numbers, ranges, label='Range', marker='o')
        # plt.plot(trial_numbers, variances, label='Variance', marker='o')
        # plt.plot(trial_numbers, iqrs, label='IQR', marker='o')
        plt.scatter(trial_numbers, trail_cog_nasa, label='Cognitive Load Score', color='black', marker='x')

        plt.xlabel('Trial Number', fontsize=14)
        plt.ylabel(f'{name_channels[channel]} Signal Statistics', fontsize=14)
        plt.legend()
        plt.gca().tick_params(labelsize=14)
        plt.show(block=False)


def plot_rms_multiple(rms_values_list, status_list, period_list):
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


def plot_data(trail):
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


def apply_notch_filters(data, fs, freqs, quality_factor=30):
    """
    Apply multiple notch filters to the EMG signal and update the EMG_signal attribute.

    Parameters:
    -----------
    data (ndarray): sensor signal data.
    freqs (list of float): Frequencies to be notched out.
    quality_factor (float): Quality factor of the notch filter.
    fs (int): Sampling frequency of the EMG signal.

    Returns:
    None: The EMG_signal attribute is updated in place.
    """
    for freq in freqs:
        b, a = iirnotch(freq, quality_factor, fs)
        data = filtfilt(b, a, data, axis=0)

    return data


def compute_band_energy(data, fs, band):
    """
    Compute the energy of a specific frequency band for each channel in the data.

    Parameters:
    data (ndarray): The input signal data.
    fs (int): The sampling frequency.
    band (tuple): The frequency band as a tuple (low, high).

    Returns:
    tuple: band_energy (ndarray) - The energy of the specified frequency band for each channel,
           Pxx_list (list) - Power Spectral Density (PSD) for each channel.
    """
    Pxx_list = []

    band_energy = np.zeros(data.shape[1])  # Initialize energy array for each channel
    for ch in range(data.shape[1]):
        f, Pxx = welch(data[:, ch], fs)
        band_mask = (f >= band[0]) & (f <= band[1])
        band_energy[ch] = np.trapz(Pxx[band_mask], f[band_mask])
        Pxx_list.append(Pxx)

    if band == (30, 124):
        return band_energy, Pxx_list

    return band_energy


# Function to segment data into overlapping windows
def segment_data(data, window_size, overlap):
    """
    :param data:
    :param window_size:
    :param overlap:
    :return:
    """
    step = int(window_size * (1 - overlap))
    segments = [data[i:i + window_size] for i in range(0, len(data) - window_size + 1, step)]
    return np.array(segments)


def plot_ratio(exp, window_size=30, overlap=0.5):
    fs = exp.fs  # Sampling frequency
    data = exp.data  # EEG data
    timestamps = exp.data_timestamps  # Timestamps for each sample
    window_size = int(window_size * fs)  # Window size of half a second
    step_size = int(window_size * (1 - overlap))  # Step size based on overlap

    # Apply fooof
    # segmented_data_with_foof = apply_fooof(data, fs, window_size, step_size)

    # Apply notch filters
    filtered_data = apply_notch_filters(data, fs, [50, 100])

    # Segment data into overlapping windows
    segmented_data = segment_data(filtered_data, window_size, overlap)

    # Frequency bands (in Hz)
    bands = {
        'EMG_PARTIAL': (51, 69),
        'EMG': (30, 124),
        'ALPHA': (8, 13),
        'BETA': (13, 30),
        'THETA': (4, 8),
        'EEG': (0.3, 30)
                        }

    # Compute energies for each segment and channel
    energies = {band: np.zeros((len(segmented_data), data.shape[1])) for band in bands}
    total_energy = np.zeros((len(segmented_data), data.shape[1]))

    Pxx_window = []

    for i, segment in enumerate(tqdm(segmented_data, desc="Processing segments")):
        for band, freq_range in bands.items():
            if band == 'EMG':
                energies[band][i], Pxx_single = compute_band_energy(segment, fs, freq_range)
                Pxx_window.append(Pxx_single)
            else:
                energies[band][i] = compute_band_energy(segment, fs, freq_range)
        total_energy[i] = np.sum([energies[band][i] for band in bands], axis=0)

    # Compute ratios
    ratios = {band: energies[band] / total_energy for band in bands}

    # ratio if beta/emg
    ratios['BETA_EMG'] = energies['BETA'] / energies['EMG']
    bands['BETA_EMG'] = (0, 124)

    # Plotting
    time_vector = np.array([timestamps[int(i * step_size + window_size / 2)] for i in range(len(segmented_data))])
    time_vector = time_vector-time_vector[0]

    for ch in range(4):
        plt.figure(figsize=(12, 6))
        for band in bands:
            if band == 'BETA_EMG':
                plt.plot(time_vector, ratios[band][:, ch], label='BETA / EMG')
            else:
                plt.plot(time_vector, ratios[band][:, ch], label=f'{band} / Total Energy')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy Ratio')
        plt.title(f'Channel {16 - ch}')
        plt.legend()
        plt.show(block=False)

    for i in range(3):
        print(np.corrcoef(ratios['EMG'][:, i], ratios['BETA'][:, i])[0, 1])

    plt.figure(figsize=(12, 6))
    plt.scatter(ratios['EMG'][:, 1], ratios['BETA'][:, 1], color='black')
    # Correctly using transform with the axis object
    ax = plt.gca()  # Get the current axis
    ax.text(0.05, 0.95, 'Channel 15',
            ha='left', va='top', transform=ax.transAxes)
    plt.xlabel('EMG Energy Ratio', fontsize=14)
    plt.ylabel('Beta Energy Ratio', fontsize=14)
    ax.tick_params(labelsize=14)
    # plt.title('EMG vs Beta Energy Ratio')
    plt.show(block=False)

    # mask = ratios['EMG'][:, 0] > 0.2
    # mask_emg = ratios['EMG'][mask, :]
    # mask_beta = ratios['BETA'][mask, :]
    #
    # for i in range(ratios['EMG'].shape[1]):
    #     corr, _ = spearmanr(mask_emg[:, i], mask_beta[:, i])
    #     print(f'Channel {16-i}: Spearman correlation', corr)
    #
    # # ratio if emg partial/emg
    # ratios['EMG_PARTIAL_EMG'] = energies['EMG_PARTIAL'] / energies['EMG']
    # bands['EMG_PARTIAL_EMG'] = (0, 124)
    #
    # plt.figure(figsize=(12, 6))
    # plt.scatter(range(len(ratios['EMG_PARTIAL_EMG'][:, 0])), ratios['EMG_PARTIAL_EMG'][:, 0])
    # plt.xlabel('3 min window')
    # plt.ylabel('EMG Partial / EMG Energy Ratio')
    # plt.title()
    # plt.show(block=False)
    #
    # # Split the ratios into 3 minute segments with overlap of 1 minute
    # num_of_values_in_sec = 4
    # num_of_values_in_3_min = 60 * num_of_values_in_sec * 3


def calculate_correlation(list_stat_trails, trail_cog_nasa):
    # Initialize an empty DataFrame to store the correlation values
    channels = list_stat_trails[0].index  # Assuming each DataFrame has the same index (channels)
    measures = list_stat_trails[0].columns.drop('name')  # Assuming each DataFrame has the same columns (measures)
    channel_names = list_stat_trails[0]['name']
    correlation_table = pd.DataFrame(index=channel_names, columns=measures)

    # Iterate over each channel (row)
    for i, channel in enumerate(channel_names):
        # Iterate over each statistical measure (column)
        for measure in measures:
            # Extract the measure values for all trails
            measure_values = np.array([df.iloc[i][measure] for df in list_stat_trails])

            # Calculate the correlation between the measure values and the subjective scores
            correlation_value = np.corrcoef(measure_values, trail_cog_nasa)[0, 1]

            # Store the correlation value in the table
            correlation_table.loc[channel, measure] = float(abs(correlation_value))

        # Create a heatmap
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(correlation_table, annot=True, cmap='coolwarm', vmin=0, vmax=1)
    # plt.title('Heatmap of Correlation between Measures and Subjective Scores')
    # plt.show()

    return correlation_table


def plot_high_correlation_graphs(list_stat_trails, trail_cog_nasa, threshold=0.7):
    # Initialize an empty DataFrame to store the correlation values
    measures = list_stat_trails[0].columns.drop('name')  # Exclude the 'name' column
    channel_names = list_stat_trails[0]['name']  # Get the channel names from the 'name' column

    # Iterate over each channel index (row)
    for i, channel in enumerate(channel_names):
        # Iterate over each statistical measure (column)
        for measure in measures:
            # Extract the measure values for all trails
            measure_values = np.array([df.iloc[i][measure] for df in list_stat_trails])

            # Calculate the correlation between the measure values and the subjective scores
            correlation_value = np.corrcoef(measure_values, trail_cog_nasa)[0, 1]

            # Check if the absolute correlation is above the threshold
            if abs(correlation_value) >= threshold:
                # Plot the objective value as a function of the subjective value
                plt.figure(figsize=(8, 6))
                plt.scatter(trail_cog_nasa, measure_values, color='black')
                plt.text(0.05, 0.95, f'{channel} - {measure}',
                         ha='left', va='top', transform=plt.gca().transAxes, fontsize=12)
                plt.xlabel('Subjective Score (NASA-TLX)', fontsize=14)
                plt.ylabel(f'Objective Measure ({measure})', fontsize=14)
                plt.tick_params(axis='both', which='major', labelsize=12)
                plt.show()


def plot_selected_graphs_as_subplots(list_stat_trails, trail_cog_nasa):
    # Define the specific channels and measures you want to plot
    plots_to_create = [
        ('EMG_ch2', 'Range'),
        ('EMG_ch5', 'Range'),
        ('EMG_ch8', 'Max'),
        ('EMG_ch8', 'Range'),
        ('EMG_ch14', 'Range'),
        ('EEG_theta_ch14', 'Range')
    ]

    # Create a 2x3 subplot
    fig, axs = plt.subplots(3, 2, figsize=(12, 16))

    # Flatten the axes array for easy iteration
    axs = axs.flatten()

    for i, (channel, measure) in enumerate(plots_to_create):
        # Find the index of the channel in the original DataFrame
        channel_index = list_stat_trails[0].index[list_stat_trails[0]['name'] == channel][0]

        # Extract the measure values for all trails
        measure_values = np.array([df.iloc[channel_index][measure] for df in list_stat_trails])

        # Calculate the correlation between the measure values and the subjective scores
        correlation_value = np.corrcoef(measure_values, trail_cog_nasa)[0, 1]

        # Plot the objective value as a function of the subjective value in the appropriate subplot
        axs[i].scatter(trail_cog_nasa, measure_values, color='black')
        axs[i].text(0.05, 0.95, f'{channel}',
                    ha='left', va='top', transform=axs[i].transAxes)
        axs[i].set_xlabel('Subjective Score (NASA-TLX)', fontsize=14)
        axs[i].set_ylabel(f'Objective Measure ({measure})', fontsize=14)

        # Customize the tick labels
        for tick in axs[i].xaxis.get_major_ticks():
            tick.label.set_fontsize(14)

        for tick in axs[i].yaxis.get_major_ticks():
            tick.label.set_fontsize(14)

    # Adjust layout for better spacing between plots
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)  # Adjust spacings

    # Adjust layout for better spacing between plots
    plt.tight_layout(pad=2.0)  # Increase pad to add space between plots

    plt.savefig('Corr', dpi=300, bbox_inches='tight')

    plt.show()


def create_plot(data, timestamps, trigger_times, triggers, title):
    # Create the plot
    channels = data.shape[1]

    fig, axs = plt.subplots(channels, 1, figsize=(10, 20), sharex=True)  # 16 rows, 1 col, shared X-axis
    fig.suptitle(title, fontsize=16)

    for i in range(channels):
        global_min = np.min(data[:, channels - 1 - i])
        global_max = np.max(data[:, channels - 1 - i])

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


def plot_spectrogram(data, fs, title):
    """
    Plot the spectrogram of the signal data.

    Args:
        data (ndarray): Signal data of shape (n_samples, n_channels).
        fs (float): Sampling frequency in Hz.
        title (str): Title for the spectrogram.
    """
    n_channels = data.shape[1]

    for i in range(n_channels):
        # Compute the spectrogram for each channel
        f_spec, t_spec, Sxx = spectrogram(data[:, i], fs)

        # Create a figure for the spectrogram
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(t_spec, f_spec, 10 * np.log10(Sxx), shading='gouraud', cmap='inferno')
        plt.colorbar(label='Power Spectral Density [dB]')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.title(f'{title} - Spectrogram (Channel {i + 1})')
        plt.tight_layout()
        plt.show()


def plot_psd_windows(data, fs, triggers, trigger_time_stamps, window_duration, title, target_trigger=22):
    """
    Plot 6 PSD windows starting from the time corresponding to the target trigger.

    Args:
        data (ndarray): Signal data of shape (n_samples, n_channels).
        fs (float): Sampling frequency in Hz.
        triggers (array): Array of trigger values.
        trigger_time_stamps (array): Array of time stamps for the triggers.
        target_trigger (int): The trigger value to start the PSD windows.
        window_duration (float): Duration of each PSD window in seconds.
        title (str): Title for the PSD windows.
    """
    n_channels = data.shape[1]
    window_samples = int(window_duration * fs)  # Samples per window

    # Find the index of the target trigger
    try:
        trigger_idx = np.where(triggers == target_trigger)[0][0]
        start_time = trigger_time_stamps[trigger_idx]
    except IndexError:
        raise ValueError(f"Target trigger {target_trigger} not found in the trigger vector.")

    for i in range(n_channels):
        # Calculate the start sample from the trigger time
        start_sample = int(start_time * fs)

        # Create a figure for the 6 PSD windows
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        axes = axes.flatten()

        for j in range(6):
            # Calculate the window start and end indices
            start_idx = start_sample + j * window_samples
            end_idx = start_idx + window_samples

            # Handle cases where the window exceeds the signal length
            if end_idx > data.shape[0]:
                start_idx = data.shape[0] - window_samples
                end_idx = data.shape[0]

            # Compute the PSD for the window
            f_psd, Pxx = welch(data[start_idx:end_idx, i], fs)

            # Plot the PSD
            ax = axes[j]
            ax.semilogy(f_psd, Pxx)
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('PSD')
            ax.set_title(f'Window {j + 1}')
            ax.grid()

        # Adjust layout and show the plots
        fig.suptitle(f'{title} - PSD Windows (Channel {i + 1})')
        plt.tight_layout()
        plt.show()