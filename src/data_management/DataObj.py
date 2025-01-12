import os
import pyintan as pyintan
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
        self.path = path
        streams, fileheader = pyxdf.load_xdf(path)
        self.file_name = os.path.basename(path)
        self.start_time = fileheader['info']['datetime']
        self.streams = streams
        self.read_lsl_streams()
        self.check_triggers()
        self.exp_data = self.read_exp_data()
        self.weights = self.read_weights()
        self.trail_cog_nasa = self.calculate_cognitive_load()
        self.sorted_indices = np.argsort(self.trail_cog_nasa)[::-1]
        self.subject_id = self.extract_subject_id()

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
        path = os.path.abspath(os.path.join(self.path, os.pardir))
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
        path = os.path.abspath(os.path.join(self.path, os.pardir))
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
        path = os.path.abspath(os.path.join(self.path, os.pardir))
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
        match = re.search(r'participant_(\d+)', self.path)
        return match.group(1).zfill(3)  # Ensures it's zero-padded to 3 digits

    def plot_nasa_tlx(self):
        """
        Plot the NASA TLX values for all the trails
        """
        plt.figure()
        plt.xlabel('Trial')
        plt.ylabel('Cognitive Load')
        plt.title('NASA TLX values for all the trails')
        plt.scatter(range(len(self.trail_cog_nasa)), self.trail_cog_nasa)
        plt.show(block=False)

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
