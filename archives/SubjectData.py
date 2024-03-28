import pyxdf
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import Tk
from scipy import signal


# Trigger ID Documentation (For Documentation Purposes Only)
# trigger_id_docs = {
#     1: 'Indicates that the participant incorrectly pressed the spacebar
#           when the stimulus did not match the N-BACK condition.',
#     2: 'Indicates that the participant pressed the spacebar correctly
#           in response to a stimulus matching the N-BACK condition.',
#     3: 'Logs instances where a response was expected (the stimulus matched the N-BACK condition)
#           but the participant did not press the spacebar, indicating a missed response.',
#     4: 'Indicates the start of a new level in the experiment.',
#     5: 'Indicates the end of the experiment.'
#     6: 'Indicates the start of the experiment.'
#     7: 'Indicates the start of the N-BACK Instructions.'
#     8: 'Indicates the end of the N-BACK Instructions.'
#     9: 'Indicates the end of a level in the experiment.',
#    10: 'Indicates when the subject has high error rates'
#    11: 'Indicates start of the NASA-TLX rating'
#    12: 'Indicates end of the NASA-TLX rating'
# }

class SubjectData:
    def __init__(self, file_path):
        self.file_path = file_path
        self.games_played = 0
        self.game_time_stamps = []
        self.data_by_type = {}
        self.data = self.load_data()
        self.filtered_data_emg = self.filter_data_emg()

    def load_data(self):
        # Load the XDF file
        streams, fileheader = pyxdf.load_xdf(self.file_path)

        # Organize data by type
        self.data_by_type = {
            'pupil_capture_fixations': [],
            'pupil_capture': [],
            'pupil_capture_pupillometry_only': [],
            'ElectrodeStream': [],
            'Trigger_Cog': []
        }

        # Extract and categorize data
        for stream in streams:
            name = stream['info']['name'][0]
            self.data_by_type[name] = {
                'info': stream['info'],
                'time_stamps': stream['time_stamps'],
                'time_series': stream['time_series']}

        # Process triggers to find the start and end of games
        self.games_played, self.game_time_stamps = self.process_triggers()

        # Segment data by game
        segmented_data = self.segment_data_by_game()

        return segmented_data

    def process_triggers(self):

        # Process triggers to find the start and end of games
        indexes_end = np.where(self.data_by_type['Trigger_Cog']['time_series'] == 9)[0]
        time_stamps_end = self.data_by_type['Trigger_Cog']['time_stamps'][indexes_end]
        indexes_start = np.where(self.data_by_type['Trigger_Cog']['time_series'] == 4)[0]
        time_stamps_start = self.data_by_type['Trigger_Cog']['time_stamps'][indexes_start]

        # Placeholder for processing logic
        games_played = len(indexes_end)
        if len(indexes_start) != games_played:
            # Remove the last start trigger
            indexes_start = time_stamps_start[:-1]
        game_time_stamps = np.column_stack((time_stamps_start, time_stamps_end))

        return games_played, game_time_stamps

    def segment_data_by_game(self):
        # Segment data by game
        segmented_data = {}

        # Segment data by game
        for i in range(len(self.game_time_stamps)):
            game_data = {}
            for key in self.data_by_type:
                # find the indices of the time stamps that fall within the game time stamps
                indexes_in_game = np.where((self.data_by_type[key]['time_stamps'] >= self.game_time_stamps[i][0]) &
                                           (self.data_by_type[key]['time_stamps'] <= self.game_time_stamps[i][1]))[0]
                # store the time stamps and time series for the data that falls within the game time stamps
                game_data[key] = {'time_stamps': self.data_by_type[key]['time_stamps'][indexes_in_game],
                                  'time_series': self.data_by_type[key]['time_series'][indexes_in_game]}
            segmented_data[i] = game_data

        return segmented_data

    def filter_data_emg(self):
        # Filter the data
        order = 5
        fs = 250
        low_cut = 0.5
        high_cut = 35
        sos = signal.butter(order, [low_cut, high_cut], btype='band', fs=fs, output='sos')
        b_notch, a_notch = signal.iirnotch(50, 20, fs)
        filtered_data = {}
        for game in range(self.games_played):
            filtered_data[game] = signal.filtfilt(b_notch, a_notch, self.data[game]['ElectrodeStream']['time_series'])
            filtered_data[game] = signal.sosfilt(sos, filtered_data[game])
        return filtered_data

    def show_data_emg(self, specific_game=[1], not_to_show=[0, 1, 2, 3, 4, 5, 6]):
        # Create a figure for each sensors data and for each sensor, create a subplot for each game
        fig, axs = plt.subplots(len(specific_game), 1, figsize=(10, 5 * len(specific_game)))
        to_show = [x for x in not_to_show if x not in specific_game]
        if len(specific_game) == 1:
            axs = [axs]
        for i, game in enumerate(specific_game):
            axs[i].plot(self.data[game]['ElectrodeStream']['time_stamps'], self.filtered_data[game][:,to_show])
            axs[i].set_title('Game ' + str(game + 1))
            axs[i].set_xlabel('Time (s)')
            axs[i].set_ylabel('EMG (mV)')
        plt.show()


# Example usage
if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    xdf_name = filedialog.askopenfilename(initialdir="raw_data", title="Select file",
                                          filetypes=(("xdf files", "*.xdf"), ("all files", "*.*")))
    root.destroy()
    subject_data = SubjectData(xdf_name)

    game_to_show = 1
    subj_emg = subject_data.filtered_data_emg[game_to_show]
    # each channel is plotted in a different subplot of 4X4
    fig, axs = plt.subplots(4, 4, figsize=(15, 15))
    for i in range(4):
        for j in range(4):
            axs[i, j].plot(subj_emg[:, i * 4 + j])
            axs[i, j].set_title('Channel ' + str(i * 4 + j + 1))
            axs[i, j].set_xlabel('Time (s)')
            axs[i, j].set_ylabel('EMG (mV)')
    plt.show()

    plt.plot(subj_emg[0:1000, 4])
    plt.title('Channel 5')
    plt.xlabel('Time (s)')
    plt.ylabel('EMG (mV)')
    plt.show()
print("Done")
