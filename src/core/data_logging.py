from src.core.screens.nasa_tlx import calculate_nasa_weight
from src.core.screens.instructions import end_experiment_screen
import os
from datetime import datetime
import csv

def log_event(trigger_id, timestamp, state, stim_flag):
    # Log event data to the variable
    state.experiment_data.append({'timestamp': timestamp, 'trigger_id': trigger_id})

    # Send the trigger to the LSL stream
    state.outlet.push_sample([trigger_id])
    print(f"Trigger {trigger_id}")
    # if stim_flag:
    #     stim_controller.start_stimulation(bursts_count=1)

def save_data_and_participant_info(state, stim_flag, user_details):
    global screen
    weights = calculate_nasa_weight()
    serial_number = user_details['Serial Number']
    folder_name = f"S{serial_number.zfill(3)}"
    project_root = os.path.join(os.path.dirname(__file__), "../../")
    folder_name = os.path.join(project_root, folder_name)
    os.makedirs(folder_name, exist_ok=True)

    # Log event for the end of the game
    log_event(5, datetime.now().timestamp(), state, stim_flag)

    # Save experiment data
    csv_file_path = os.path.join(folder_name, 'experiment_data.csv')
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['timestamp', 'trigger_id'])
        writer.writeheader()
        for data in state.experiment_data:
            writer.writerow(data)

    # Save participant details
    details_file_path = os.path.join(folder_name, 'participant_details.txt')
    with open(details_file_path, 'w') as file:
        for key, value in user_details.items():
            file.write(f"{key}: {value}\n")

    # Save error performance and NASA-TLX weights together
    error_file_path = os.path.join(folder_name, 'stage_performance.csv')
    with open(error_file_path, 'w', newline='') as file:
        # Include weight keys in the fieldnames
        fieldnames = list(state.stage_performance[0].keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        # Assuming weights should be included in every line of stage_performance data
        for data in state.stage_performance:
            writer.writerow(data)

    # Save NASA-TLX weights
    weights_file_path = os.path.join(folder_name, 'nasa_tlx_weights.csv')
    with open(weights_file_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=weights.keys())
        writer.writeheader()
        writer.writerow(weights)

    # End an experiment definitively
    end_experiment_screen(state)