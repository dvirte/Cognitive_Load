from DataObj import DataObj
from ExpObj import ExpObj
from tkinter import filedialog
from tkinter import Tk


subj_name = 'dvir'
subj_n = '002'
root = Tk()
root.withdraw()
root.attributes('-topmost', True)
xdf_name = filedialog.askopenfilename(initialdir="raw_data", title="Select file",
                                      filetypes=(("xdf files", "*.xdf"), ("all files", "*.*")))

data = DataObj(xdf_name)
diocan = ExpObj(data.ElectrodeStream, data.Trigger_Cog)
status = 1  # 0 for rest, 1 for task
period = 30  # represents the period of the task

# Plot the data before filtering
cali_before = diocan.extract_emg_trials(status, period)
diocan.plot_trail(cali_before)

# Plot the data after notch filter of 50 Hz
diocan.apply_notch_filters([50, 100])
diocan.apply_bandpass_filter(lowcut=4, highcut=120)
cali_after_50 = diocan.extract_emg_trials(status, period)
diocan.plot_trail(cali_after_50)
diocan.plot_data(cali_after_50)

a=5