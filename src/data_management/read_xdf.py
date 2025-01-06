import pyxdf
from tkinter import filedialog
from tkinter import Tk

# Open window to select file from 'raw_data' folder
root = Tk()
root.withdraw()
root.attributes('-topmost', True)
xdf_name = filedialog.askopenfilename(initialdir = "raw_data", title = "Select file", filetypes = (("xdf files","*.xdf"),("all files","*.*")))
root.destroy()

# Load the file
streams, fileheader = pyxdf.load_xdf(xdf_name)

info = []
time_stamps = []
time_series = []
name = []

for stream in streams:
    info.append(stream['info'])
    time_stamps.append(stream['time_stamps'])
    time_series.append(stream['time_series'])
    name.append(stream['info']['name'])

print(info)