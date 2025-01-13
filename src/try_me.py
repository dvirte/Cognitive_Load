from src.data_management.DataObj import DataObj
from src.visualization import vis_cog as vc
from src.core.ExpProcessor import ExpProcessor
from src.signal_processing.emd_blinking import plot_emd_signal
# from emd_blinking import plot_emd_signal, segment_data
# from jaw_stiffness import extract_features
import pandas as pd
import pickle
import os
from src.signal_processing import proccess_func as pf



# Current path
current_path = os.getcwd()

id_num = '07'

# Merge the paths
if 'src' in current_path:
    current_path = os.path.join(current_path, '..')
xdf_name = os.path.join(current_path, 'data', f'participant_{id_num}', 'S01', '01.xdf')


# find the subject ID
subject_id = xdf_name.split('\\')[-2]

data = DataObj(xdf_name)

# Session folder
session_folder = os.path.dirname(xdf_name)

# Load the data if not already loaded
if f'processor_{id_num}.pkl' not in os.listdir(session_folder):

    # Initialize ExpProcessor
    processor = ExpProcessor(
        emg_data=data.ElectrodeStream,
        trigger_stream=data.Trigger_Cog,
        fs=250,
        window_size=30.0,
        overlap=0.5,
        subject_id=data.subject_id,  # Pass the subject ID
        sorted_indices=data.sorted_indices,  # Pass the sorted indices,
        auto_process=False,
        path = xdf_name
        )

    # Save the object to a file with pickle and subject
    with open(rf'{session_folder}\processor_{id_num}.pkl', 'wb') as f:
        pickle.dump(processor, f)

else:
    # Load the object from the file
    with open(rf'{session_folder}\processor_{id_num}.pkl', 'rb') as f:
        processor = pickle.load(f)
    f.close()


# Initialize the processor instance
maze_rank = 26

# Plot the EMG signal for the specified maze
plot_emd_signal(processor, maze_rank)

# Process the data for calibration
pf.barin_process(processor, 2, 2)

# Extract rate of challenging tasks
sorted_indices = data.sorted_indices

# # Extract the most challenging task
# pf.barin_process(processor, 1, sorted_indices[0])

# pf.plot_alpha_calibration(processor)

vc.plot_signal_with_time(processor)


plot_list = [[2,1],[1,33],[1,34],[1,35],[0,35],[0,36],[0,37]]
for i, j in plot_list:
    print(f'Plotting status {i} and period {j}')
    trial = processor.extract_trials(status=i,period=j)
    vc.plot_trail(trial)


# Plot the FOOOF results for the specified maze
processor.plot_fooof(top_n_windows=6, indx=39, least_tow=True) # lightest task with the least 2 windows
processor.plot_fooof(top_n_windows=6, indx=0, least_tow=False) # 0 for the most challenging maze


# Extract jaw features for the specified maze
extract_features(processor, maze_rank)

# diocan = ExpObj(data.ElectrodeStream, data.Trigger_Cog, fs=250)

# # Plot the NASA TLX values for all the trails
# data.plot_nasa_tlx()

# # Plot the first 10 minutes of the data
# data.plot_first_10_minutes()

# vc.plot_ratio(diocan)

# status =    # 0 for rest, 1 for task, 2 for calibration
# period =    # represents the period of the task. arbitrary number for calibration

# # Trying FOOOF
# res = fe.apply_FOOOF(data)


# Plot spectrogram of the first channel
if 1 == 0:
    trail_33_emg = diocan.extract_band_trials(1, 33, 'EMG')
    trail_33_beta = diocan.extract_band_trials(1, 33, 'EEG_beta')
    diocan.plot_multiple_channels(trail_33_emg, trail_33_beta, ['EMG', 'EEG_beta'], [13, 14, 15])
    trail_34_emg = diocan.extract_band_trials(1, 34, 'EMG')
    trail_34_beta = diocan.extract_band_trials(1, 34, 'EEG_beta')
    trail_35_emg = diocan.extract_band_trials(1, 35, 'EMG')
    trail_35_beta = diocan.extract_band_trials(1, 35, 'EEG_beta')
    trail_36_emg = diocan.extract_band_trials(1, 36, 'EMG')
    trail_36_beta = diocan.extract_band_trials(1, 36, 'EEG_beta')
    trail_37_emg = diocan.extract_band_trials(1, 37, 'EMG')
    trail_37_beta = diocan.extract_band_trials(1, 37, 'EEG_beta')


# Plot the EMG of hardest task
hardest_task = data.sorted_indices[0]-1
task_plot = diocan.extract_band_trials(1, hardest_task, 'EMG')
# vc.plot_trail(task_plot)

# Plot the EMG of medium task
if 1 == 0:
    for i in [0, 1, 2, 15, 20, 28, 37]:
        rms = diocan.stat_band(1, i, stat='rms', relevant=False)
        vc.plot_stat(rms, 1, i, 'RMS', ['EMG', 'EMG_complementary'])

list_stat_trails = []
for i in range(0, len(data.sorted_indices) - 1):
    rms = diocan.stat_band(1, i, stat='rms', relevant=True)
    list_stat_trails.append(diocan.extract_statistics(rms))

# Assuming you have your list_stat_trails and data.trail_cog_nasa ready
correlation_results = vc.calculate_correlation(list_stat_trails, data.trail_cog_nasa[:-1])

# # Plot ratio between EMG and beta
# vc.plot_ratio(diocan)
#
# # Plot Yael graph as subplot
# vc.plot_selected_graphs_as_subplots(list_stat_trails, data.trail_cog_nasa[:-1])
#
# # Plot all high correlation graphs
# vc.plot_high_correlation_graphs(list_stat_trails, data.trail_cog_nasa[:-1])

# Plot the statistics of all channels
vc.plot_channel_statistics(list_stat_trails, data.trail_cog_nasa[:-1])


input("Press Enter to close the plots and exit...")
