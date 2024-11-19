from psychopy import visual, core, sound, parallel, event
import sounddevice as sd
import numpy as np
import random
import os
import csv
from datetime import datetime
from XtrRT_master.XtrRT_master.XtrRT.data import Data

# ==========================
# Setup
# ==========================

# Get the current directory
current_dir = os.getcwd()

# Main experiment folder
experiment_folder = os.path.join(current_dir, 'Jaw_Movement_Experiment')

# Create the main experiment folder if it doesn't exist
if not os.path.exists(experiment_folder):
    os.makedirs(experiment_folder)

# Get list of existing subject folders
existing_subjects = [
    name for name in os.listdir(experiment_folder)
    if os.path.isdir(os.path.join(experiment_folder, name)) and name.isdigit()
]

# Determine next subject number
if existing_subjects:
    subject_numbers = [int(name) for name in existing_subjects]
    next_subject_number = max(subject_numbers) + 1
else:
    next_subject_number = 1

# Format subject number as two-digit string
subject_number = f"{next_subject_number:02d}"

# Create subject folder
subject_folder = os.path.join(experiment_folder, subject_number)
os.makedirs(subject_folder, exist_ok=True)

# Start data recording
host_name = "127.0.0.1"
port = 20001
n_bytes = 1024
data = Data(host_name, port, verbose=False, timeout_secs=15,
            save_as=f"Jaw_Movement_Experiment/{subject_number}/{subject_number}.edf")
data.start()

data.add_annotation("Start recording")

# Prepare data storage
data_triggers = []

# Create a window
win = visual.Window(fullscr=True, color='black')

# Parameters for the beep sound
frequency = 440  # Frequency of the beep in Hz (A4 note)
duration = 0.5   # Duration of the beep in seconds
sample_rate = 44100  # Sample rate (samples per second)

# Generate a sine wave for the beep
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
beep = 0.5 * np.sin(2 * np.pi * frequency * t)

# Initial instruction
initial_instruction = visual.TextStim(
    win,
    text="In order to start the experiment, press Enter.",
    color='white'
)
initial_instruction.draw()
win.flip()
event.waitKeys(keyList=['return'])

# Start the experiment clock
experiment_clock = core.Clock()

# Define movements and corresponding trigger codes
movements = ['Clenching', 'Jaw opening', 'Chewing', 'Resting']
movement_triggers = {
    'Clenching': 1,
    'Jaw opening': 2,
    'Chewing': 3,
    'Resting': 4
}

# Create a list of movements, each repeated 15 times
movement_list = movements * 15  # Total of 60 trials
random.shuffle(movement_list)   # Randomize the order

# ==========================
# Main Experiment Loop
# ==========================

for movement in movement_list:
    # --------------------------
    # Break Period (5 seconds)
    # --------------------------
    instruction_text = f"When the sound is heard, please do: {movement}"
    instruction = visual.TextStim(win, text=instruction_text, color='white')
    instruction.draw()
    win.flip()
    core.wait(5)  # Display the instruction for 5 seconds

    # --------------------------
    # Action Period (5 seconds)
    # --------------------------
    # Play the beep sound
    sd.play(beep, samplerate=sample_rate)

    # Send trigger if parallel port is set up
    trigger_time = experiment_clock.getTime()  # Time in seconds since experiment started
    trigger_value = movement_triggers[movement]
    data.add_annotation(str(trigger_value))  # Add trigger to the data stream
    print(f"Trigger for '{movement}' (code {trigger_value}) would be sent here.")

    # Record the data
    data_triggers.append({
        'Movement': movement,
        'Trigger': trigger_value,
        'Time': trigger_time
    })

    # Wait for the participant to perform the action
    core.wait(5)

# ==========================
# Save Data
# ==========================

# Define the filename
filename = f"Jaw_Movement_Experiment_{subject_number}.csv"
filepath = os.path.join(subject_folder, filename)

# Write data to CSV
with open(filepath, mode='w', newline='') as csv_file:
    fieldnames = ['Movement', 'Trigger', 'Time']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    for entry in data_triggers:
        writer.writerow(entry)

print(f"Data saved to {filepath}")

# ==========================
# End of Experiment
# ==========================

end_instruction = visual.TextStim(
    win,
    text="The experiment is over. Thank you very much. Please do not press any key.",
    color='white',
    wrapWidth=1.5  # Adjust text wrapping if necessary
)
end_instruction.draw()
win.flip()
event.waitKeys(keyList=['return'])

# ==========================
# Cleanup
# ==========================

data.add_annotation("Stop recording")
data.stop()

print(data.annotations)
print('process terminated')

# Close the window and exit
win.close()
core.quit()