import os
import pyintan as pyintan
import numpy as np
import scipy
from load_intan_rhd_format import read_data
import pandas as pd
import glob
import pyxdf as pyxdf



class DataObj:
    def __init__(self, path, reduce_faulty_electrodes=None):
        self.path = path
        self.output_folder = self.create_output_folder(path)
        streams, fileheader = pyxdf.load_xdf(path)
        self.file_name = os.path.basename(path)
        self.start_time = fileheader['info']['datetime']
        self.streams = streams
        self.read_lsl_streams()
        self.check_triggers()
        self.exp_data = self.read_exp_data()

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
            return data_df.to_dict(orient='list')
        except FileNotFoundError:
            print('No stage_performance.csv file found')
            return None
