import pyxdf
streams, fileheader = pyxdf.load_xdf('sub-P000_ses-S000_task-Default_run-001_meg.xdf')

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