import os
# import pyintan as pyintan
import numpy as np
import scipy
from intanutil.load_intan_rhd_format import read_data
import pandas as pd
import glob
import pyxdf as pyxdf
import csv
import matplotlib.pyplot as plt
import re


class DataObj:
    def __init__(self, path, reduce_faulty_electrodes=None):
        """
       Initialize DataObj with one or more XDF files.

       Args:
           path_or_paths: Single path string or list of path strings to XDF file(s)
           reduce_faulty_electrodes: Optional parameter for electrode filtering
       """
        if isinstance(path, str):
            self.path = [path]
            self.single_path = path
        else:
            self.path = path
            self.single_path = path[0]

        # Sort paths to ensure correct chronological order (01.xdf before 02.xdf)
        self.path.sort()

        # Store first file's name and metadata
        self.file_name = os.path.basename(self.path[0])

        # Load and merge all XDF files
        self.load_and_merge_xdf_files()

        # Process the data
        self.check_triggers()
        self.exp_data = self.read_exp_data()
        self.weights = self.read_weights()
        self.trail_cog_nasa = self.calculate_cognitive_load()
        self.sorted_indices = np.argsort(self.trail_cog_nasa)[::-1]
        self.subject_id = self.extract_subject_id()

    def load_and_merge_xdf_files(self):
        """
        Load and merge multiple XDF files maintaining chronological order
        and resolving stream overlaps.
        """
        all_streams = []

        # Track the first file's timestamp for offset calculation
        self.start_time = None

        # Load each XDF file
        for i, path in enumerate(self.path):
            print(f"Loading XDF file {i + 1}/{len(self.path)}: {os.path.basename(path)}")
            streams, fileheader = pyxdf.load_xdf(path)

            # Store source file information in each stream for later reference
            for stream in streams:
                stream['source_file'] = os.path.basename(path)

            # Set the start time from the first file
            if i == 0:
                self.start_time = fileheader['info']['datetime']
                self.streams = streams  # Store initial streams
                all_streams.extend(streams)
            else:
                # For subsequent files, calculate time offset relative to first file
                # This assumes files are in chronological order
                time_offset = self._calculate_time_offset(self.streams, streams)
                print(f"Applied time offset of {time_offset:.4f}s to file {os.path.basename(path)}")

                # Adjust timestamps in the new streams by adding the offset
                for stream in streams:
                    # Store original first timestamp for debugging
                    original_first_ts = stream['time_stamps'][0]

                    # Apply the offset
                    stream['time_stamps'] += time_offset

                    # Log the adjustment for key streams
                    if 'name' in stream['info'] and stream['info']['name'][0] in ['ElectrodeStream', 'Trigger_Cog']:
                        adjusted_first_ts = stream['time_stamps'][0]
                        print(f"  → {stream['info']['name'][0]}: First timestamp adjusted from "
                              f"{original_first_ts:.4f}s to {adjusted_first_ts:.4f}s")

                all_streams.extend(streams)

        # Merge streams with the same name
        self.streams = self._merge_streams_by_name(all_streams)

        # Print summary of merged streams
        print("\nMerged streams summary:")
        for stream in self.streams:
            name = stream['info']['name'][0]
            n_samples = len(stream['time_stamps'])
            time_span = stream['time_stamps'][-1] - stream['time_stamps'][0]
            print(f"  → {name}: {n_samples} samples spanning {time_span:.2f}s")

            # If this stream has file transitions, print them
            if 'file_transitions' in stream:
                print("    File transitions:")
                for trans in stream['file_transitions']:
                    print(f"      {trans['file_name']}: {trans['start_time']:.2f}s - {trans['end_time']:.2f}s "
                          f"({trans['end_time'] - trans['start_time']:.2f}s duration)")

        # Read the merged LSL streams into instance attributes
        self.read_lsl_streams()

    def _calculate_time_offset(self, first_streams, second_streams):
        """
        Calculate the time offset between two sets of streams.

        Strategy:
        1. Find common stream types in both files
        2. Use the last timestamp of first file and first timestamp of second file
        3. Handle both overlap and gap cases to create a continuous signal

        Returns:
            time_offset: Time in seconds to add to second file timestamps
        """
        # Get stream names from both files
        first_names = [stream['info']['name'][0] for stream in first_streams]
        second_names = [stream['info']['name'][0] for stream in second_streams]

        # Find common streams
        common_streams = set(first_names).intersection(set(second_names))

        if not common_streams:
            # If no common streams, use the max timestamp from first file
            max_time = max([stream['time_stamps'][-1] for stream in first_streams])
            return max_time

        # Use the first common stream to calculate offset
        stream_name = 'ElectrodeStream'

        # Find the stream in first file
        first_stream = next(s for s in first_streams if s['info']['name'][0] == stream_name)
        second_stream = next(s for s in second_streams if s['info']['name'][0] == stream_name)

        # Get the last timestamp from first file and first timestamp from second file
        last_ts_first = first_stream['time_stamps'][-1]
        first_ts_second = second_stream['time_stamps'][0]

        # Check if there's a gap or overlap
        if last_ts_first < first_ts_second:
            # Gap case: Make the second file start immediately after the first file
            # Calculate how much to shift the second file's timestamps
            time_shift = last_ts_first - first_ts_second

            # Add a small gap (equivalent to one sample) for continuity
            # Estimate the sampling rate from the first few samples
            if len(first_stream['time_stamps']) > 1:
                sample_period = first_stream['time_stamps'][1] - first_stream['time_stamps'][0]
                time_shift += sample_period  # Add one sample period for continuity

            return time_shift
        else:
            # Overlap case: find where second file should actually start
            # We'll set it so second file's first timestamp matches first file's last timestamp
            return last_ts_first - first_ts_second

    def _merge_streams_by_name(self, all_streams):
        """
        Merge streams with the same name from different files.

        Args:
            all_streams: List of all streams from all files

        Returns:
            merged_streams: List of merged streams
        """
        # Group streams by name
        stream_groups = {}
        for stream in all_streams:
            name = stream['info']['name'][0]
            if name not in stream_groups:
                stream_groups[name] = []
            stream_groups[name].append(stream)

        # Merge each group
        merged_streams = []
        for name, streams in stream_groups.items():
            if len(streams) == 1:
                # If only one stream with this name, no merging needed
                merged_streams.append(streams[0])
            else:
                # Merge multiple streams with the same name
                merged_stream = streams[0].copy()  # Start with the first stream

                # Initialize lists for concatenation
                all_time_stamps = list(streams[0]['time_stamps'])
                all_time_series = list(streams[0]['time_series'])

                # Add data from other streams, ensuring timestamps are in order
                for stream in streams[1:]:
                    # Get timestamps and time series data from the current stream
                    time_stamps = stream['time_stamps']
                    time_series = stream['time_series']

                    # Skip duplicate timestamps (potential overlap between files)
                    # Find first timestamp that's greater than the last timestamp in our merged stream
                    start_idx = 0
                    while start_idx < len(time_stamps) and time_stamps[start_idx] <= all_time_stamps[-1]:
                        start_idx += 1

                    if start_idx < len(time_stamps):
                        # Check if we need to fill a gap between the files
                        if start_idx == 0 and time_stamps[0] > all_time_stamps[-1]:
                            # There's a gap between the files - we already adjusted timestamps
                            # in _calculate_time_offset, but might need to handle the transition
                            # smoothly here as well

                            # For visualization purposes, we'll add a marker in the merged stream
                            # to indicate a file transition (can be modified based on needs)
                            print(f"File transition at timestamp {time_stamps[0]:.2f}")

                        # Append non-overlapping data
                        all_time_stamps.extend(time_stamps[start_idx:])
                        all_time_series.extend(time_series[start_idx:])

                # Convert lists back to numpy arrays
                merged_stream['time_stamps'] = np.array(all_time_stamps)
                merged_stream['time_series'] = np.array(all_time_series)

                # Store info about the number of samples from each source file
                if 'file_transitions' not in merged_stream:
                    merged_stream['file_transitions'] = []

                # Keep track of sample counts from each file
                current_pos = 0
                for i, stream in enumerate(streams):
                    n_samples = len(stream['time_stamps'])
                    file_name = f"File {i + 1}"
                    if 'source_file' in stream:
                        file_name = stream['source_file']

                    # Store transition points
                    merged_stream['file_transitions'].append({
                        'file_index': i,
                        'file_name': file_name,
                        'start_idx': current_pos,
                        'end_idx': current_pos + n_samples - 1,
                        'start_time': merged_stream['time_stamps'][current_pos],
                        'end_time': merged_stream['time_stamps'][min(current_pos + n_samples - 1,
                                                                     len(merged_stream['time_stamps']) - 1)]
                    })
                    current_pos += n_samples

                merged_streams.append(merged_stream)

        return merged_streams

    @staticmethod
    def create_output_folder(path):
        output_folder = path.split(os.path.sep)[-1][:-4]
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        return output_folder

    def read_lsl_streams(self):
        for stream in self.streams:
            stream_name = stream['info']['name'][0]
            current_time_stamps = np.array(stream['time_stamps'])
            current_time_series = np.array(stream['time_series'])

            if hasattr(self, stream_name):
                existing_stream = getattr(self, stream_name)

                # Concatenate time stamps and time series
                existing_stream['time_stamps'] = np.concatenate((existing_stream['time_stamps'], current_time_stamps))
                existing_stream['time_series'] = np.concatenate((existing_stream['time_series'], current_time_series),
                                                                axis=0)

            else:
                # Create a new attribute for this stream
                setattr(self, stream_name, {
                    'time_stamps': current_time_stamps,
                    'time_series': current_time_series
                })

    def read_triggers_csv(self):
        path = os.path.abspath(os.path.join(self.single_path, os.pardir))
        file_path = os.path.join(path, 'experiment_data.csv')

        try:
            data_df = pd.read_csv(file_path)
            return data_df.to_dict(orient='list')
        except FileNotFoundError:
            print('No experiment_data.csv file found')
            return None

    def check_triggers(self):
        triggers_reference = self.read_triggers_csv()
        triggers_xdf = self.Trigger_Cog['time_series'][:, 0]

        try:
            triggers_csv = np.array(triggers_reference['trigger_id'])

            # Find the index of the first matching trigger in the CSV data
            match_index = np.where(triggers_csv == triggers_xdf[0])[0][0]

            if match_index > 0:  # If there are missed triggers
                missed_triggers_values = triggers_csv[:match_index].reshape(-1, 1)
                diff_time = self.Trigger_Cog['time_stamps'][0] - triggers_reference['timestamp'][match_index]
                missed_triggers_times = np.array(triggers_reference['timestamp'][:match_index]) + diff_time

                # Prepend missed triggers to the XDF data
                self.Trigger_Cog['time_series'] = np.vstack((missed_triggers_values, self.Trigger_Cog['time_series']))
                self.Trigger_Cog['time_stamps'] = np.concatenate((missed_triggers_times, self.Trigger_Cog['time_stamps']))

        except IndexError:
            return None

    def read_exp_data(self):
        path = os.path.abspath(os.path.join(self.single_path, os.pardir))
        file_path = os.path.join(path, 'stage_performance.csv')

        try:
            data_df = pd.read_csv(file_path)
            data_df = data_df.to_dict(orient='list')
            # Change values of dictionary to numpy arrays
            for key in data_df.keys():
                data_df[key] = np.array(data_df[key])
            data_df['Performance'] = 19 - data_df['Performance']
            return data_df
        except FileNotFoundError:
            print('No stage_performance.csv file found')
            return None

    def read_weights(self):
        path = os.path.abspath(os.path.join(self.single_path, os.pardir))
        file_path = os.path.join(path, 'nasa_tlx_weights.csv')
        with open(file_path, mode='r') as file:
            csv_reader = csv.reader(file)
            headers = next(csv_reader)  # This is the first row with headers
            weights = next(csv_reader)  # This is the second row with weights

            # Convert weights from strings to integers
            weights = list(map(int, weights))
            total_weight = sum(weights)

            # Normalize the weights
            normalized_weights = {headers[i]: weights[i] / total_weight for i in range(len(weights))}
            return normalized_weights

    def calculate_cognitive_load(self):
        # Calculate the cognitive load based on the NASA-TLX weights
        cognitive_loads = np.zeros(len(self.exp_data['Effort']))
        for key in self.weights.keys():
            cognitive_loads += self.weights[key] * self.exp_data[key]
        return cognitive_loads

    def extract_subject_id(self):
        """
        Extracts the subject ID from the filename.
        Assumes filename contains 'sub-PXXX' where XXX is the subject number.
        """
        match = re.search(r'participant_(\d+)', self.single_path)
        return match.group(1).zfill(3)  # Ensures it's zero-padded to 3 digits

    def plot_nasa_tlx(self, show_figure = True):
        """
        Plot the NASA TLX values for all the trails

        Returns:
        - fig: The matplotlib figure object for saving
        """
        fig = plt.figure()
        plt.xlabel('Trial')
        plt.ylabel('Cognitive Load')
        plt.title('NASA TLX values for all the trails')
        plt.scatter(range(len(self.trail_cog_nasa)), self.trail_cog_nasa)
        if show_figure:
            plt.show(block=False)
        return fig

    def plot_first_10_minutes(self):

        # Get the first 10 minutes of the data
        ind_10_min = np.where(self.ElectrodeStream['time_stamps'] <
                              self.ElectrodeStream['time_stamps'][0] + 10 * 60)[0][-1]

        time_series = self.ElectrodeStream['time_series'][:ind_10_min, :]
        time_stamps = self.ElectrodeStream['time_stamps'][:ind_10_min]

        # Get the trigger times that are within the first 10 minutes
        trigger_times = self.Trigger_Cog['time_stamps']
        lsat_sample = time_stamps[-1]
        last_trigger_index = np.where(trigger_times < lsat_sample)[0][-1]
        trigger_times = trigger_times[:last_trigger_index]
        first_sample = time_stamps[0]

        # Adjust the trigger times to be relative to the first sample
        trigger_times = trigger_times - first_sample
        time_stamps = time_stamps - first_sample

        plt.figure(figsize=(15, 5))
        plt.plot(time_stamps, time_series[:, 0])

        # Add vertical red lines for triggers
        for trigger_time in trigger_times:
            plt.axvline(x=trigger_time, color='r')

        plt.title('First Channel - First 10 Minutes')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show(block=False)
