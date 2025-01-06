import mne
import numpy as np
import matplotlib.pyplot as plt
import copy
import pywt
from scipy.interpolate import griddata
import cv2
import matplotlib.image as mpimg
from numpy.linalg import inv
import os
from classifying_ica_components.classifying_ica_components import (perform_ica_algorithm,
                                                                   classify_participant_components_using_atlas,
                                                                   extract_and_order_ica_data)
from picard import picard

from classifying_ica_components.classifying_ica_components import plot_ica_heatmap


class EDFFile:
    def __init__(self, file_path):
        """
        Initialize the EDFFile object by reading the EDF file.

        Parameters:
        - file_path: str
            The path to the EDF file.
        """

        # current directory
        current_directory = os.getcwd()
        #add the data folder to the path
        data_folder = r"data"
        self.data_path = fr"{current_directory}\{data_folder}"


        # Get the participant ID and session number from the file path
        self.participant_ID = file_path.split('/')[:-1][1]

        # Session folder path. go back one folder and then go to the session folder
        self.session_folder_path = '/'.join(file_path.split('/')[:-1])

        # Session number
        self.session_number = file_path.split('/')[:-1][2]

        # Read the EDF file and preload the data for faster access
        self.raw = mne.io.read_raw_edf(file_path, preload=True)

        # Get the signal data (channels x samples)
        self.signal = self.raw.get_data()
        # Remove the IMU data
        self.signal = self.signal[0:16, :]
        # Normalize the signal
        self.signal = (self.signal - np.mean(self.signal, axis=1, keepdims=True)) / \
                      np.std(self.signal, axis=1, keepdims=True)


        # Get the timeline of each sample in seconds
        self.times = self.raw.times

        # Convert annotations to events array and event_id dictionary
        self.events, self.event_id = mne.events_from_annotations(self.raw)

        # Map event codes to event names for easy lookup
        self.event_mapping = {event_code: event_name for event_name, event_code in self.event_id.items()}

        # Get the event times in seconds
        self.event_times = self.times[self.events[:,0]]

        # Create the label vector
        self.labels = self.create_label_vector()

        # Aplly ICA to the data
        atlas_folder = r"classifying_ica_components\atlas"
        self.im_dir = 'face_ica.jpeg' # path to the image
        self.x_coor_path = f"{atlas_folder}\side_x_coor.npy"
        self.x_coor = np.load(self.x_coor_path) # load x coordinates
        self.y_coor_path = f"{atlas_folder}\side_y_coor.npy"
        self.y_coor = np.load(self.y_coor_path) # load y coordinates
        self.wavelet = 'db15' # wavelet type
        self.threshold = np.load(f"{atlas_folder}/threshold.npy")
        self.centroids_lst = []
        for i in range(17):
            current_centroid = np.load(f"{atlas_folder}/cluster_{i + 1}.npy")
            current_centroid = np.nan_to_num(current_centroid, nan=0)
            self.centroids_lst.append(current_centroid)
        experinment_part_name ="" # name of the experiment part
        if not os.path.exists(fr'{self.session_folder_path}\{self.participant_ID}_{self.session_number}_heatmap_{self.wavelet}.npy'):
            perform_ica_algorithm(file_path, self.participant_ID, self.session_number, self.session_folder_path,
                                self.im_dir, self.x_coor, self.y_coor, 16)
        participant_data = np.load(fr'{self.session_folder_path}\{self.participant_ID}_{self.session_number}_heatmap_{self.wavelet}.npy')
        participant_data = np.nan_to_num(participant_data, nan=0)
        classify_participant_components_using_atlas(participant_data, self.data_path, self.centroids_lst,
                                                    self.participant_ID, self.session_folder_path, self.session_number,
                                                    self.threshold, self.wavelet, experinment_part_name,
                                                    self.im_dir, self.participant_ID, self.session_folder_path,
                                                    self.x_coor, self.y_coor)
        self.Y = extract_and_order_ica_data(self.participant_ID, self.session_folder_path, self.session_number)

        # # calculate ICA
        # self.K, self.W, self.Y = picard(self.signal)
        # self.plot_ica_heatmap()

        # # Get the normalized signal
        # self.normalized_signal = self.normalize_by_event()

    def create_label_vector(self):
        """
        Create a label vector for the signal based on event annotations.

        Returns:
        - labels: np.ndarray
            A vector of labels for the entire signal (one label per sample).
        """
        # Initialize the label vector with zeros
        labels = np.zeros(self.signal.shape[1], dtype=int)

        # Process each event
        for i, event in enumerate(self.events):
            event_sample_idx = event[0]  # Event onset in sample indices
            event_label = event[2]  # Event code (label)

            # Skip labels outside the range 1–4
            if event_label not in range(1, 5):
                continue

            # Get the event time in seconds
            event_time = self.times[event_sample_idx]

            # Find all indices within 5 seconds after the event
            end_time = event_time + 5  # 5 seconds after the event
            time_mask = (self.times >= event_time) & (self.times <= end_time)

            # Assign the label to the corresponding samples
            labels[time_mask] = event_label

        return labels

    def split_events(self, train_ratio=0.8):
        """
        Split events into train and test sets based on their labels (1-4 only).

        Parameters:
        - train_ratio: float
            Proportion of events to use for training (default: 80%).

        Returns:
        - train_events: np.ndarray
            Events assigned to the training set.
        - test_events: np.ndarray
            Events assigned to the testing set.
        """
        # Extract events with labels 1-4
        labeled_events = self.events[np.isin(self.events[:, 2], range(1, 5))]

        train_events = []
        test_events = []

        # Split events by label
        for label in range(1, 5):
            label_events = labeled_events[labeled_events[:, 2] == label]
            np.random.shuffle(label_events)
            split_point = int(len(label_events) * train_ratio)

            train_events.extend(label_events[:split_point])
            test_events.extend(label_events[split_point:])

        return np.array(train_events), np.array(test_events)

    def label_windows(self, train_events, test_events, window_size=0.3):
        """
        Divide the signal into windows and assign labels based on events and train-test split.

        Parameters:
        - train_events: np.ndarray
            Events assigned to the training set.
        - test_events: np.ndarray
            Events assigned to the testing set.
        - window_size: float
            Duration of each window in seconds.

        Returns:
        - window_labels: np.ndarray
            Labels for each window (positive labels for train/test, 0 for ambiguous).
        - window_split: np.ndarray
            Indicates whether each window belongs to train (1) or test (0).
        """
        sampling_rate = int(self.raw.info['sfreq'])  # Sampling rate in Hz
        samples_per_window = int(window_size * sampling_rate)
        num_windows = self.Y.shape[1] // samples_per_window

        # Initialize window labels and split indicators
        window_labels = np.zeros(num_windows, dtype=int)
        window_split = np.zeros(num_windows, dtype=int)

        # Iterate over each window
        for window_idx in range(num_windows):
            start_sample = window_idx * samples_per_window
            end_sample = (window_idx + 1) * samples_per_window

            # Extract samples in the current window
            window_samples = self.labels[start_sample:end_sample]

            # Determine the dominant label in the window
            unique, counts = np.unique(window_samples, return_counts=True)
            label_counts = dict(zip(unique, counts))

            # Find the label with the highest count
            max_label = max(label_counts, key=label_counts.get)
            max_count = label_counts[max_label]

            # Assign the label if >80% of the samples match the label
            if max_label != 0 and max_count / samples_per_window > 0.8:
                window_labels[window_idx] = max_label
                # Check proximity to train or test events (forward only)
                train_events_after = [event[0] for event in train_events if
                                      start_sample >= event[0] and start_sample <= event[0] + 5 * sampling_rate]
                test_events_after = [event[0] for event in test_events if
                                     start_sample >= event[0] and start_sample <= event[0] + 5 * sampling_rate]

                if train_events_after:
                    # Assign to train
                    window_split[window_idx] = 1  # Train

        return window_labels, window_split

    def extract_features(self, window_indices, window_size=0.3, wavelet='db7'):
        """
        Extract features for the given window indices.

        Parameters:
        - window_indices: np.ndarray
            Indices of the windows to extract features for.
        - window_size: float
            Duration of each window in seconds.
        - wavelet: str
            Wavelet type for feature extraction.

        Returns:
        - features: np.ndarray
            Matrix of extracted features, where each row corresponds to a window.
        """
        sampling_rate = int(self.raw.info['sfreq'])  # Sampling rate in Hz
        samples_per_window = int(window_size * sampling_rate)

        # List to store extracted features for each window
        all_features = []

        for window_idx in window_indices:
            # Determine the sample range for the current window
            start_sample = window_idx * samples_per_window
            end_sample = start_sample + samples_per_window

            # Extract the signal for the current window
            window_signal = self.signal[:, start_sample:end_sample]

            # Compute wavelet coefficients (CD4, CD5)
            cd4, cd5 = self.calculate_cd(window_signal, wavelet)

            # Extract features (MAV, STD, etc.)
            features = self.calculate_features(cd4, cd5, window_signal)

            # Flatten features into a single vector for this window
            window_features = np.hstack([np.hstack(f) for f in features])
            all_features.append(window_features)

        # Convert the list of features to a numpy array
        return np.array(all_features)

    def normalize_by_event(self):
        """
        Normalize the signal by the mean activity value for each label (event).

        Normalization adjusts the signal for inter-subject variability by dividing
        the signal during each event (label) by the mean value of the signal for that event.

        Returns:
        - normalized_signal: np.ndarray
            The event-normalized signal with the same shape as `self.signal`.
        """
        # Get the unique labels (events) from the label vector
        unique_labels = np.unique(self.labels)

        # Create a copy of the signal to normalize
        normalized_signal = np.copy(self.signal)

        for label in unique_labels:
            if label == 0:
                # Skip label 0 (unlabeled or insignificant events)
                continue

            # Find the indices corresponding to the current label
            event_indices = np.where(self.labels == label)[0]

            if len(event_indices) > 0:
                event_mean = np.mean(self.signal[:, event_indices], axis=1, keepdims=True)
                normalized_signal[:, event_indices] /= (event_mean + 1e-8)  # Avoid division by zero

        return normalized_signal

    def plot_signal(self, type='ALL'):
        """
        Plot the raw signal data.
        """

        # Define movements and corresponding trigger codes
        movement_triggers = {
            1: 'Clenching',
            2: 'Jaw opening',
            3: 'Chewing',
            4: 'Resting',
            5: 'Start of trial',
            6: 'End of trial',
        }

        trigger_times = self.event_times
        triggers = self.events[:, 2]

        # Determine the duration of the recording
        total_duration = self.times[-1]  # in seconds

        # Define the segment duration (5 minutes = 300 seconds)
        segment_duration = 60  # seconds

        # Calculate the number of segments
        num_segments = int(np.ceil(total_duration / segment_duration))

        if type == 'ALL':
            selected_channels = self.signal
            channels = 16
        elif type == 'EEG':
            selected_channels = self.signal[12:16, :]
            channels = 4
        elif type == 'JAW':
            selected_channels = self.signal[0:3, :]
            channels = 3


        # Plot each segment
        for segment in range(num_segments):
            # Determine the start and end times for this segment
            start_time = segment * segment_duration
            end_time = min((segment + 1) * segment_duration, total_duration)

            # Find the indices corresponding to the start and end times
            start_idx = np.searchsorted(self.times, start_time)
            end_idx = np.searchsorted(self.times, end_time)

            # Extract the time and signal data for this segment
            timestamps = self.times[start_idx:end_idx]
            signal_segment = self.Y[:, start_idx:end_idx]

            # Find triggers within this segment
            segment_trigger_indices = np.where((trigger_times >= start_time) & (trigger_times < end_time))[0]
            segment_trigger_times = trigger_times[segment_trigger_indices]
            segment_triggers = triggers[segment_trigger_indices]

            fig, axs = plt.subplots(nrows=channels, ncols=1, figsize=(30, 20), sharex=True)

            title = f"Raw EMG Signal - Segment {segment + 1}/{num_segments} ({start_time:.1f}s to {end_time:.1f}s)"
            fig.suptitle(title, fontsize=16)

            # Plot original EMG signal
            global_min = np.min(signal_segment[0:channels, :])
            global_max = np.max(signal_segment[0:channels, :])

            for i in range(channels):
                axs[i].plot(timestamps, signal_segment[channels-i-1,:])  # Plot each channel
                axs[i].text(1.02, 0.5, f'EMG {channels - i}', transform=axs[i].transAxes, va='center', ha='left')
                axs[i].spines['top'].set_visible(False)  # Hide the top spine
                axs[i].spines['right'].set_visible(False)  # Hide the right spine
                axs[i].spines['left'].set_visible(False)  # Optionally, hide the left spine as well
                axs[i].spines['bottom'].set_visible(False)  # Hide the bottom spine
                axs[i].get_xaxis().set_visible(False)  # Hide x-axis labels and ticks
                axs[i].set_ylim(global_min, global_max)  # Set the same Y-axis limits for all subplots

                # Add vertical red lines for triggers
                for trigger_time in segment_trigger_times:
                    axs[i].axvline(x=trigger_time, color='red', linestyle='--')

            # Add trigger labels on the last subplot
            for j, trigger_time in enumerate(segment_trigger_times):
                axs[-1].text(trigger_time, axs[-1].get_ylim()[0] - 0.05 * (axs[-1].get_ylim()[1] - axs[-1].get_ylim()[0]),
                             f'{movement_triggers[segment_triggers[j]]}', color='red', ha='center', va='top', rotation=0)

            # Adjust settings for the last subplot
            axs[-1].spines['bottom'].set_visible(True)  # Show the bottom spine for the last subplot
            axs[-1].get_xaxis().set_visible(True)  # Show x-axis labels and ticks for the last subplot
            axs[-1].set_xlabel("Time [s]")  # Common X-axis title
            fig.text(0.04, 0.5, 'Amplitude [mV]', va='center', rotation='vertical')  # Common Y-axis title

            plt.subplots_adjust(hspace=0)  # Remove horizontal space between plots
            plt.show(block=False)

            # Save the figure
            fig.savefig(f"raw_emg_segment_{segment + 1}.png")

            # Close the figure to avoid memory issues
            plt.close()

    def jaw_features(self, window_size=0.3, wavelet='db7'):

        # Prepare to store features for each channel
        list_of_features = [[], [], [], [], []]  # MAV_CD4, STD_CD4, MAV_CD5, STD_CD5, MAV_x
        channel_features = []

        # Define the number of samples in each window
        num_of_samples = int(window_size * self.raw.info['sfreq'])

        for i in range(16): # 3 channels
            channel_features.append(copy.deepcopy(list_of_features))

        # Loop through the EMG signal in windows
        for i in range(0, self.Y.shape[1], num_of_samples):
            emg_window = self.Y[:, i:i + num_of_samples]

            # Calculate CD4 and CD5 coefficients
            cd4, cd5 = self.calculate_cd(emg_window, wavelet)

            # Calculate features from CD4, CD5, and the raw signal
            features = self.calculate_features(cd4, cd5, emg_window)

            # Append to the list of features
            for j in range(len(features)):
                for channel in range(16):
                    val = features[j][channel]
                    (channel_features[channel][j].append(val))

        # Plot the statistics
        self.plot_statistic(channel_features, window_size)

    def calculate_cd(self, emg_segment, wavelet='db7'):
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
        for channel_idx in range(emg_segment.shape[0]):  # Iterate over 3 channels
            channel_data = emg_segment[channel_idx, :]

            # Perform Discrete Wavelet Transform (DWT)
            coeffs = pywt.wavedec(channel_data, wavelet, level=5)

            # Extract CD4 and CD5 correctly
            cd5 = coeffs[1]  # CD5 is at level 5
            cd4 = coeffs[2]  # CD4 is at level 4

            # Append to results
            cd4_all_channels.append(cd4)
            cd5_all_channels.append(cd5)

        return cd4_all_channels, cd5_all_channels

    def calculate_features(self, cd4, cd5, raw_signal):
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

    def plot_statistic(self, features, window_size=0.3):

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

            # Save the figure
            fig.savefig(f"channel_{channel + 1}_statistics.png")

    def plot_ica_heatmap(self, number_of_channels=16):
        image, height, width = self.image_load()  # load image
        # calculations for the heatmap
        inverse = np.absolute(inv(self.W))
        grid_y, grid_x = np.mgrid[1:height + 1, 1:width + 1]
        points = np.column_stack((self.x_coor, self.y_coor))
        f_interpolate = []
        for i in range(number_of_channels):
            interpolate_data = griddata(points, inverse[:, i], (grid_x, grid_y), method='linear')
            norm_arr = self.norm(interpolate_data)
            f_interpolate.append(norm_arr)  # plot heatmap
        np.save(fr"_heatmap", f_interpolate)
        fig, axs = plt.subplots(2, int(number_of_channels / 2), figsize=(16, 8), dpi=300)
        axs = axs.ravel()
        fig.subplots_adjust(hspace=0, wspace=0.01)
        for i in range(number_of_channels):
            axs[i].imshow(image)
            axs[i].pcolormesh(f_interpolate[i], cmap='jet', alpha=0.5)
            axs[i].set_title("ICA Source %d" % (i + 1))
            axs[i].axis('off')
        plt.suptitle(f"ICA components heatmaps")
        plt.savefig(fr"_heatmap.png")
        plt.close()

    def image_load(self):
        image_path = self.im_dir
        # load the image, write the path where the image is saved (if there is no image uncomment these two lines)
        img = plt.imread(image_path)
        # convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = mpimg.imread(image_path)  # for heatmap

        # image dimensions
        height = img.shape[0]
        width = img.shape[1]

        return image, height, width

    def norm(self, arr):
        myrange = np.nanmax(arr) - np.nanmin(arr)
        norm_arr = (arr - np.nanmin(arr)) / myrange
        return norm_arr
