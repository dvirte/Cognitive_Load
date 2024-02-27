# Import necessary libraries
from psychopy import visual, event, core
from application import Application
from ble import scan_ble_devices, upload_application, setLevel, connect_ble, disconnect_ble, impedance_check, play, stop
import asyncio
import numpy as np
import sounddevice as sd
import os
import glob
import random
# import pylsl
import os
import pylsl
outlet = pylsl.StreamOutlet(pylsl.StreamInfo("MyTriggerStream", "Markers", 1, 0, pylsl.cf_int32, "myuidw43536"))
trigger_marker = 1  # You can use any integer value as a trigger code
 #todo
 # add LSL triggers of specific stimulation used for phosphene
 # shuffle the stimulation order in stimulation_extended
 # add the condition of repeating trial if you found waldo before the stimulation started
 # check reaction time of waldo
using_controller = False
# Create an LSL outlet
# outlet = pylsl.StreamOutlet(pylsl.StreamInfo("MyTriggerStream", "Markers", 1, 0, pylsl.cf_int32, "myuidw43536"))

device_name = "BrainPatch2-0003"
data_dir = "data/"
stim_mode = {}
stim_mode['constant'] = 1
stim_mode['negative_constant'] = -1
stim_mode['sine'] = 2
stim_mode['square'] = 3
stim_mode['ramp'] = 4
stim_mode['negative_ramp'] = -4
stim_mode['sin_like'] = 5
stim_mode['pulse'] = 6
stim_mode['interphase_gap'] = 8
stim_mode['biphasic_ramp'] = 9
stim_mode['balanced_pulse'] = 10
stim_mode['pulsed_sin'] = 11
stim_mode['balanced_sin'] = 12

trigger_list = {
    'play_stimulus': 1,
    'stop_stimulus': 2,
    'mouse_left_press': 3,
    'mouse_right_press': 4,
    'mouse_left_press': 3,
    'mouse_right_press': 4,
    'waldo_panel_on': 5,
    'waldo_panel_off': 6,
    'fixation_panel_on': 7,
    'fixation_panel_off': 8,
    'stimulation_panel_on':9,
    'stimulation_panel_off':10,
    'question_panel_on':11,
    'question_panel_off':12,
    'space_press': 13,
    'enter_press': 14,
    'escape_press': 15,
}



stimulations = [
    {
        'stim_mode_stim': 8,
        'freq_stim': 1,
        'comfort_level': 100,
        'duty_size_stim': 10,
        'threshold_multiplier': 0.2,
        'duration_stim': 1000
    },
    {
        'stim_mode_stim': 2,
        'freq_stim': 20,
        'comfort_level': 150,
        'duty_size_stim': 10,
        'threshold_multiplier': 1,
        'duration_stim': 1000
    },
    {
        'stim_mode_stim': 3,
        'freq_stim': 1,
        'comfort_level': 100,
        'duty_size_stim': 49,
        'threshold_multiplier': 0.1,
        'duration_stim': 1000
    }
]

total_duration = 2

async def main():
    async def upload_and_run(app_local, comfort_level):
        await upload_application(app_local)
        await setLevel(comfort_level)
        play_beep()
        outlet.push_sample([trigger_list['play_stimulus']])
        await play()

        # enable one or the other, otherwise it goes straight to disconnect and stops the stimulation
        # await asyncio.sleep(total_duration)
        impedance, current, voltage = await impedance_check(total_duration)
        await stop()
        outlet.push_sample([trigger_list['stop_stimulus']])
        play_beep()
        return impedance, current, voltage

    def play_beep():
        # Duration of the beep sound in seconds
        duration = 0.1

        # Frequency of the beep sound in Hz (adjust as needed)
        beep_freq = 500

        # Sample rate and buffer size
        sample_rate = 44100
        buffer_size = 1024

        # Generate a beep sound signal
        t = np.linspace(0, duration, int(duration * sample_rate), False)
        beep_signal = np.sin(2 * np.pi * beep_freq * t)

        # Play the beep sound
        sd.play(beep_signal, samplerate=sample_rate, blocksize=buffer_size)
        sd.wait()

    def update_gaussian_cursor(mouse, gaussian):
        x, y = mouse.getPos()
        buttons = mouse.getPressed()
        if buttons[0]:  # If left mouse button is clicked
            # Confirm the choice and proceed with the experiment
            return True
        # Update the Gaussian size based on the roll of the mouse
        wheel_rel = mouse.getWheelRel()[1]
        if wheel_rel != 0:  # If there is wheel movement
            current_size = gaussian.size[0]  # Assuming the Gaussian is a square, size[0] == size[1]
            new_size = current_size + wheel_rel * 0.01  # Adjust the multiplier as needed for sensitivity
            new_size = max(min(new_size, 0.2), 0.01)  # Enforce minimum and maximum size
            gaussian.size = (new_size, new_size)  # Update the size
        gaussian.pos = (x, y)
        return False  # Continue updating the cursor

    # Create a window in fullscreen mode
    win = visual.Window(size=(1920, 1080),fullscr=True, color='gray')

    # Hide the cursor
    win.mouseVisible = False

    # Create a text stimulus
    text = visual.TextStim(win, text='Welcome to the experiment', color='black', height=0.1)

    # Connect to the device outside the loop
    if using_controller:
        await scan_ble_devices(device_name)
        await connect_ble()

    # Initialize comfort level

    stimulation_index = 0
    # Main experiment loop
    escape_pressed = False  # Track if escape key is pressed
    space_pressed = False
    First_panel = True
    Second_panel = True

    stimulation_index = 0
    comfort_level_0 = 250
    comfort_level = comfort_level_0
    multip_0 = 1
    multip = multip_0
    mouse = event.Mouse(visible=False)
    while not escape_pressed:  # Continue until escape key is pressed
        # Check for key press

        value_to_find = stimulations[stimulation_index]['stim_mode_stim']
        for mode, value in stim_mode.items():
            if value == value_to_find:
                actual_stim_mode = mode
        keys = event.getKeys(keyList=['space', 'escape', 'return'])
        while not space_pressed and First_panel:
            keys = event.getKeys(keyList=['space', 'escape'])
            text.text = f'Welcome to the experiment\n'
            text.draw()
            win.flip()
            if 'space' in keys:
                space_pressed = True
                First_panel = False
            if 'escape' in keys:
                escape_pressed = True
                break
        if Second_panel:
            space_pressed = False
        while not space_pressed and Second_panel:
            keys = event.getKeys(keyList=['space', 'escape'])
            text.text = f'Starting {actual_stim_mode} stimulation protocol\n'
            comfort_level = stimulations[stimulation_index]['comfort_level']
            multip = stimulations[stimulation_index]['threshold_multiplier']
            text.draw()
            win.flip()
            if 'space' in keys:
                space_pressed = True
                Second_panel = False
            if 'escape' in keys:
                escape_pressed = True
                break
        if 'space' in keys:
            outlet.push_sample([trigger_list['space_press']])
            app = Application(stim_freq=stimulations[stimulation_index]['stim_mode_stim'],
                              duty_size=stimulations[stimulation_index]['duty_size_stim'],
                              stim_mode=stimulations[stimulation_index]['stim_mode_stim'],
                              zero_wave=0, point_index=0, multiplier=multip, duration=stimulations[stimulation_index]['duration_stim'])
            if using_controller:
                impedance, current, voltage = await upload_and_run(app, comfort_level)

                # Access the elements of the tuples
                impedance_value = impedance[0]
                current_value = current[0]
                voltage_value = voltage[0]

                # Create a string with the values to display
                display_text = f'Comfort Level: {np.round(multip * comfort_level, 1)}\nImpedance: {impedance_value:.2f} kOhm\nCurrent: {current_value:.2f} mA\nVoltage: {voltage_value:.2f} V'

                # Draw the text on the screen
                text.text = f'\n{display_text}'
                text.draw()
            win.flip()

            # Increase comfort level for the next press
            multip += 0.05  # Adjust the increment value as needed
        if 'return' in keys:
            outlet.push_sample([trigger_list['enter_press']])
            # Store the current comfort level for the current stimulation
            stimulations[stimulation_index]['threshold_multiplier'] = multip
            # Move to the next stimulation
            stimulation_index += 1
            multip = multip_0
            Second_panel = True
            space_pressed = False
            if stimulation_index >= len(stimulations):
                escape_pressed = True  # Exit the loop if all stimulations are completed
                break

        if 'escape' in keys:
            outlet.push_sample([trigger_list['escape_press']])
            escape_pressed = True  # Exit the loop if escape key is pressed

    # Disconnect the device and show the cursor again
    # await disconnect_ble()
    image_folder = 'generated_images/'

    # Use glob to find all image files in the folder
    image_paths = glob.glob(os.path.join(image_folder, '*.png'))  # Assumes the images are in PNG format
    num_images_per_quadrant = 15
    image_filenames = []
    for quadrant in range(4):
        for i in range(num_images_per_quadrant):
            filename = f'generated_images/waldo_quadrant_{quadrant + 1}_image_{i + 1}.png'
            image_filenames.append(filename)
    random.shuffle(image_filenames)
    # Create the stimulations list for 60 images
    stimulations_extended = []
    num_trials = num_images_per_quadrant * 4
    for i in range(num_trials):
        # Alternate between the three sets of parameters
        stimulation = stimulations[i % 3].copy()
        stimulation['waldopic'] = image_filenames[i]  # Assign a random image filename
        stimulations_extended.append(stimulation)

    # Print the stimulations list
    for i, stimulation in enumerate(stimulations_extended, start=1):
        print(f"Stimulation {i}: {stimulation}")

    # Initialize a list to store trial data
    trial_data = []

    # Create a window
    # win = visual.Window(fullscr=False, color='gray')

    # Create a fixation dot
    fixation = visual.Circle(win, radius=0.01, fillColor='black', pos=(0, 0))

    # Load the "Waldo" image

    # Create a mouse object for tracking mouse position


    # Create a clock to measure time
    trial_clock = core.Clock()
    search_panel_finished = False
    # Main experiment loop
    win.mouseVisible = True

    # Create text stimuli for the question and response options
    question_text = visual.TextStim(win, text='Did you perceive a flickering?', pos=(0, 0.2), color='black')
    yes_text = visual.TextStim(win, text='Yes', pos=(-0.2, -0.2), color='black')
    no_text = visual.TextStim(win, text='No', pos=(0.2, -0.2), color='black')
    time_fixation = 1

    for trial in range(num_trials):
        trial_info = {}
        # Present the fixation dot for 2 seconds with random jitter
        fixation.draw()
        win.flip()
        outlet.push_sample([trigger_list['fixation_panel_on']])
        jitter_time = random.uniform(0, 0.25)
        jitter_time_flash = random.uniform(2, 3)
        # Start measuring trial duration
        trial_clock.reset()
        while trial_clock.getTime() < time_fixation + jitter_time:
            ciao = 1
        # await scan_ble_devices(device_name)
        # await connect_ble()
        app = Application(stim_freq=stimulations_extended[trial]['stim_mode_stim'],
                                                    duty_size=stimulations_extended[trial]['duty_size_stim'],
                                                    stim_mode=stimulations_extended[trial]['stim_mode_stim'],
                                                    zero_wave=0, point_index=0, multiplier=stimulations_extended[trial]['threshold_multiplier'], duration=stimulations_extended[trial]['duration_stim'])
        if using_controller:
            impedance, current, voltage = await upload_and_run(app, stimulations_extended[trial]['threshold_multiplier'])
        outlet.push_sample([trigger_list['fixation_panel_off']])
        outlet.push_sample([trigger_list['question_panel_on']])
        question_text.draw()
        yes_text.draw()
        no_text.draw()
        win.flip()
        # Wait for a response (mouse click)
        response_time = None
        while response_time is None:
            if mouse.getPressed()[0]:  # Left mouse click
                outlet.push_sample([trigger_list['mouse_left_press']])
                response_time = trial_clock.getTime()
                yes_text.color = 'red'  # Change the color of "Yes" to white
                yes_text.draw()
                no_text.draw()
                question_text.draw()
                win.flip()
                core.wait(0.5)  # Display for 0.5 seconds
                yes_text.color = 'black'  # Reset the color
                trial_info['fixation_response'] = True
            elif mouse.getPressed()[2]:  # Right mouse click
                outlet.push_sample([trigger_list['mouse_right_press']])
                response_time = trial_clock.getTime()
                no_text.color = 'red'  # Change the color of "No" to white
                yes_text.draw()
                no_text.draw()
                question_text.draw()
                win.flip()
                core.wait(0.5)  # Display for 0.5 seconds
                no_text.color = 'black'  # Reset the color
                trial_info['fixation_response'] = False

        # Start measuring trial duration
        trial_clock.reset()

        # Present the "Waldo" image (search panel) and wait for a response
        # You can customize the position and size of the image as needed
        # Load and display the current Waldo image
        image_path = stimulations_extended[trial]['waldopic']
        waldo_image = visual.ImageStim(win, image=image_path, pos=(0, 0))
        waldo_image.draw()
        win.flip()

        # Replace the following with your code for getting responses
        search_panel_presented = True  # Change to True when search panel is presented
        response_time = None  # Initialize response time
        stimulation_flag = False
        while search_panel_presented:
            if trial_clock.getTime() > jitter_time_flash and not stimulation_flag:
                if using_controller:
                    impedance, current, voltage = await upload_and_run(app,stimulations_extended[trial]['threshold_multiplier'])
                stimulation_flag = True
            # Check for a response and record the time when the response occurs
            if response_time is None:
                if mouse.getPressed()[0]:
                    response_time = trial_clock.getTime()
                    search_panel_finished = True
                    # Get and store the mouse position
                    mouse_x, mouse_y = mouse.getPos()
                    trial_info['mouse_position'] = (mouse_x, mouse_y)

            # Replace this condition with your code to check if the search panel is finished
            if response_time is not None and search_panel_finished:
                search_panel_presented = False
        core.wait(0.1)
        question_text.draw()
        yes_text.draw()
        no_text.draw()
        win.flip()
        # Wait for a response (mouse click)
        response_time = None
        while response_time is None:
            if mouse.getPressed()[0]:  # Left mouse click
                response_time = trial_clock.getTime()
                yes_text.color = 'red'  # Change the color of "Yes" to white
                yes_text.draw()
                no_text.draw()
                question_text.draw()
                win.flip()
                core.wait(0.5)  # Display for 0.5 seconds
                yes_text.color = 'black'  # Reset the color
                trial_info['waldo_response'] = True
            elif mouse.getPressed()[2]:  # Right mouse click
                response_time = trial_clock.getTime()
                no_text.color = 'red'
                yes_text.draw()
                no_text.draw()
                question_text.draw()
                # Change the color of "No" to white
                win.flip()
                core.wait(0.5)  # Display for 0.5 seconds
                no_text.color = 'black'  # Reset the color
                trial_info['waldo_response'] = False
        # Create a Gaussian blob as the cursor
        # Create a Gaussian blob as the cursor
        gaussian_cursor = visual.GratingStim(win, tex='gauss', mask='circle', size=0.3,
                                             pos=(0, 0), sf=0, color='white')
        circle = visual.Circle(win, radius=0.5, fillColor='black', lineColor='black')
        # Create a mouse object
        mouse = event.Mouse(win=win)
        continue_experiment = True
        while continue_experiment:
            circle.draw()
            gaussian_cursor.draw()
            win.flip()
            # Update the Gaussian cursor based on mouse input
            if update_gaussian_cursor(mouse, gaussian_cursor):
                continue_experiment = False

                # Calculate trial duration, fixation duration, and reaction time
        trial_duration = trial_clock.getTime()
        fixation_duration = 2 + jitter_time
        reaction_time = response_time - fixation_duration if response_time is not None else None

        # Store trial data in the dictionary
        trial_info['trial_number'] = trial + 1
        trial_info['trial_duration'] = trial_duration
        trial_info['fixation_duration'] = fixation_duration
        trial_info['reaction_time'] = reaction_time

        # Append the trial data to the list
        trial_data.append(trial_info)
        print(f"Trial {trial + 1} Data: {trial_info}")
        # Check for the escape key to exit the experiment
        keys = event.getKeys(keyList=['escape'])
        if 'escape' in keys:
            break
        np.save(f'{data_dir}results.npy', trial_data)
    # Print or save the participant's responses

    # ... (rest of the code)
    # Close the window and end the experiment
    if using_controller:
        await disconnect_ble()
    win.close()
    core.quit()

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())



# Function to update the Gaussian cursor position and radius based on mouse movement and clicks

