import os
import random
import json
import numpy as np
import pygame
import copy
from datetime import datetime
from data_logging import log_event
from screens.instructions import instruction_screen
from screens.nasa_tlx import nasa_tlx_rating_screen
from calibration.sync_calibration import synchronization_calibration


def create_maze_background(maze, cell_size, BLACK, WHITE):
    maze_background = pygame.Surface((maze.shape[1] * cell_size, maze.shape[0] * cell_size))
    maze_background.fill(BLACK)
    for y in range(maze.shape[0]):
        for x in range(maze.shape[1]):
            if maze[y, x] == 1:
                pygame.draw.rect(maze_background, WHITE, (x * cell_size, y * cell_size, cell_size, cell_size))
    return maze_background


def load_maze_from_file(dim, path_of_maze):
    """
    Load a random maze from the pre-generated files based on dimension.
    """
    dir_path = os.path.join(os.path.dirname(__file__), f"../../resources/ready_maze/dim{dim}")
    maze_files = [f for f in os.listdir(dir_path) if f.endswith('.json')]
    random_maze_file = random.choice(maze_files)
    maze_path = os.path.join(dir_path, random_maze_file)
    path_of_maze.append(maze_path)
    with open(maze_path, 'r') as file:
        maze_array = json.load(file)
    return np.array(maze_array)  # Convert back to a numpy array if needed


def setup_level(state, cfg, adjust_levels=True):

    if adjust_levels:

        if state.baseline_maze < cfg.amount_of_levels + 2:
            state.baseline_maze += 1
            # Structured increase in difficulty for the training phase
            if state.baseline_maze <= 4:
                cfg.maze_size = 5  # First 4 rounds with maze size 5
            elif 5 <= state.baseline_maze <= 7:
                cfg.maze_size = 10  # Next 3 rounds with maze size 10
            elif 8 <= state.baseline_maze <= 10:
                cfg.maze_size = 15  # Last 3 rounds with maze size 15
            else:
                cfg.maze_size = 5  # Default to 5 for rounds beyond the specified training phase
        if state.baseline_maze > 1:
            # Analyze the latest performance from experiment_data
            # Placeholder for performance analysis logic
            error_rate = analyze_performance(state)
            state.stage_performance.append({'timestamp': datetime.now().timestamp(), 'error_rate': error_rate,
                                      'n_back_level': state.n_back_level, 'maze_size': cfg.maze_size,
                                      'animal_sound': state.animal_sound,
                                      "path_of_maze": copy.deepcopy(state.path_of_maze),
                                      'sound_delay': cfg.sound_delay})
            state.path_of_maze = []
            print(f"Performance error: {1-error_rate}")

            # Logic to decide on increasing difficulty or ending the experiment
            if 1-error_rate <= 0.2 and state.baseline_maze == cfg.amount_of_levels+2:
                # Decide which aspect to increase: This is a simplified example; implement your own logic
                if state.level_list[0] == 0:  # Increase the N-back level
                    state.animal_sound = True
                    state.n_back_level += 1
                    state.level_list[0] = 1
                elif state.level_list[1] == 0:  # Increase the maze size
                    cfg.maze_size = min(cfg.maze_size + 5, 50)
                    state.level_list[1] = 1
                elif state.level_list[2] == 0:  # Increase the maze complexity
                    state.level_list[2] = 1
                else:
                    state.animal_sound = False
                    state.level_list = [0, 0, 0]  # Reset the level list

                if 1-error_rate <= 0.05:
                    # Decrease the delay between sounds by 0.5 seconds, down to the minimum
                    cfg.sound_delay = max(cfg.sound_delay - 500, cfg.min_sound_delay)
                    print("sound delay decreased")

            elif 1-error_rate >= 0.6 and state.baseline_maze == cfg.amount_of_levels+2:
                # send trigger to indicate that the subject has high error rates
                log_event(10, datetime.now().timestamp(), state, cfg.stim_flag)
                # Increase the delay between sounds by 0.5 seconds, up to the maximum
                cfg.sound_delay = min(cfg.sound_delay + 500, cfg.max_sound_delay)
                print("sound delay increased")

    # Adjusted code to load the maze instead of generating a new one
    maze = load_maze_from_file(cfg.maze_size, state.path_of_maze)

    # Calculate the size of each cell to fit the screen
    maze_width = maze.shape[1]
    maze_height = maze.shape[0]
    cell_size = min(state.screen_width // maze_width, state.screen_height // maze_height)

    # Calculate the offsets to center the maze
    offset_x = (state.screen_width - maze_width * cell_size) // 2
    offset_y = (state.screen_height - maze_height * cell_size) // 2

    # Create the maze background for the new level
    maze_background = create_maze_background(maze, cell_size, cfg.BLACK, cfg.WHITE)

    return maze, cell_size, maze_background, offset_x, offset_y


def analyze_performance(state):

    # Initialize counts of levels completed
    state.levels_completed = sum(data['trigger_id'] == 4 for data in state.experiment_data)

    # Calculation of the amount of triggers of each type for the level that is now finished, and the previous level
    current_triggers = [0, 0, 0]

    state.performance_ratios['end'].append(datetime.now().timestamp())

    for data in reversed(state.experiment_data):
        if data['trigger_id'] == 4:
            state.performance_ratios['start'].append(data['timestamp'])
            break
        if data['trigger_id'] < 4:
            current_triggers[data['trigger_id'] - 1] += 1

    state.performance_ratios['TP'].append(current_triggers[1])
    state.performance_ratios['FP'].append(current_triggers[0]+current_triggers[2])

    # Total number of reports
    total_reports = sum(current_triggers)

    # Calculate the correctness percentage
    try:
        correct_percentage = current_triggers[1] / total_reports
    except ZeroDivisionError:
        return 1

    return correct_percentage


def is_maze_completed(player_x, player_y, maze):  # check if the player reached the exit
    # Check if player reached the exit Assuming exit is at [-2, -1]
    if player_x == maze.shape[1] - 1 and player_y == maze.shape[0] - 2:
        return True
    else:
        return False


def complete_maze(state, cfg, user_details):

    # Reset the key state to avoid unintended movement
    state.key_pressed = None  # Reset key state

    # Calculate the elapsed time for the current maze
    elapsed_time = (datetime.now() - state.stage_start_time).total_seconds()

    if (datetime.now() - state.experiment_start_time).total_seconds() > cfg.time_of_experiment*60: # experiment ended
        # Call the rating screen function after a maze is completed
        error_rate = analyze_performance(state)
        state.stage_performance.append({
            'timestamp': datetime.now().timestamp(),
            'error_rate': error_rate,  # Ensure this is updated
            'n_back_level': state.n_back_level,
            'maze_size': cfg.maze_size,
            'animal_sound': state.animal_sound,
            "path_of_maze": copy.deepcopy(state.path_of_maze),
            'sound_delay': cfg.sound_delay,
        })
        nasa_tlx_rating_screen(state, cfg.stim_flag)

        # Call the synchronization calibration
        synchronization_calibration(state, cfg.stim_flag)
        # experiment ended

        state.running = False  # Set running to False to exit the game loop
        return  # End the function

    elif elapsed_time < 60 and state.baseline_maze >= cfg.amount_of_levels + 2:  # Less than 1 minute
        # Log event indicating a new maze is started due to quick completion
        log_event(20, datetime.now().timestamp(), state, cfg.stim_flag)  # Trigger ID 20

        # Set up a new maze with the same parameters without changing levels or showing rating screen
        state.maze, state.cell_size, state.maze_background, state.offset_x, state.offset_y =  (
            setup_level(state, cfg, adjust_levels=False))

    else:
        # Set up the new level
        state.maze, state.cell_size, state.maze_background, state.offset_x, state.offset_y = (
            setup_level(state, cfg, adjust_levels=True))

        # Call the rating screen function after a maze is completed
        nasa_tlx_rating_screen(state, cfg.stim_flag)

        # Add calibration synchronization after half of the experiment time has passed
        if ((datetime.now() - state.experiment_start_time).total_seconds() > (cfg.time_of_experiment * 60)/2 and
                state.middle_calibration):
            synchronization_calibration(state, cfg.stim_flag)
            state.middle_calibration = False

        # Reset the sound sequence for the new level
        state.sound_sequence = []

        # Change the screen size back to instruction screen size
        instruction_screen_set = pygame.display.set_mode((state.screen_width, state.screen_height))

        # Call the instruction screen with the updated window size
        if not instruction_screen(state, cfg.stim_flag, user_details):
            exit()  # Exit if the user closes the window or presses ESCAPE


        # Record the start time of the new stage
        state.stage_start_time = datetime.now()  # Add this line

    # Log event for the start of a new level
    if state.experiment_data[-1]['trigger_id'] != 20: # If the last trigger was not for a new maze in the same level
        log_event(4, datetime.now().timestamp(), state, cfg.stim_flag)

    # Adjust the    screen dimensions to fit the new maze
    state.screen = pygame.display.set_mode((state.screen_width, state.screen_height))

    # Reset player position and other necessary variables for the new level
    state.player_x, state.player_y = 0, 1  # Reset player position to the start of the maze

