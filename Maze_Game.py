import pygame
import os
import random
import numpy as np
import pandas as pd
import csv
from datetime import datetime
from create_maze import maze_to_ndarray
import pylsl
import json

# Initialize LSL Stream
outlet = pylsl.StreamOutlet(pylsl.StreamInfo("MyTriggerStream", "Markers", 1, 0, pylsl.cf_int32, "myuidw43536"))

# Initialize Pygame and the mixer
pygame.init()
pygame.mixer.init()

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

experiment_data = []  # Initialize an empty list to store event data
maze_size = 5  # Initial maze size
maze_complexity = 0.8  # Initial maze complexity
animal_sound = False  # The n-back task only takes into account if the sound played is from animals and not inanimate
level_list = [0, 0, 0]  # List to keep track of which aspect to increase next
performance_ratios = {'TP': [], 'FP': [], 'start': [], 'end': []}  # Initialize performance ratios
stage_performance = []  # Initialize list to store high error performance


def log_event(trigger_id, timestamp):
    global experiment_data
    # Log event data to the variable
    experiment_data.append({'timestamp': timestamp, 'trigger_id': trigger_id})

    # Send the trigger to the LSL stream
    outlet.push_sample([trigger_id])


def display_text(screen, text, position, font_size=50, color=(255, 255, 255)):
    font = pygame.font.Font(None, font_size)
    text_surface = font.render(text, True, color)
    rect = text_surface.get_rect(center=position)
    screen.blit(text_surface, rect)


def draw_text(surface, text, position, font_size=32, color=(255, 255, 255), center=False):
    font = pygame.font.Font(None, font_size)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    if center:
        text_rect.center = position
    else:
        text_rect.topleft = position
    surface.blit(text_surface, text_rect)


def input_screen_old():
    user_details = {'Serial Number': ''}
    current_input = 'Serial Number'
    input_active = True

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if input_active:
                    if event.key == pygame.K_RETURN:
                        if current_input == 'Serial Number':
                            current_input = 'Age'
                        elif current_input == 'Age':
                            current_input = 'Gender'
                        else:
                            return user_details  # Finished input
                    elif event.key == pygame.K_BACKSPACE:
                        user_details[current_input] = user_details[current_input][:-1]
                    elif event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                        user_details['Gender'] = 'Female' if user_details['Gender'] == 'Male' else 'Male'
                    else:
                        if current_input != 'Gender':  # Allow typing for non-gender fields
                            user_details[current_input] += event.unicode

        screen.fill((0, 0, 0))
        draw_text(screen, "Enter User Details", (320, 50), center=True)
        draw_text(screen, f"Serial Number: {user_details['Serial Number']}", (320, 150), center=True)
        draw_text(screen, f"Age: {user_details['Age']}", (320, 200), center=True)
        draw_text(screen, f"Gender: {user_details['Gender']} (Use Left/Right to change)", (320, 250), center=True)

        pygame.display.flip()
        pygame.time.Clock().tick(30)


def input_screen():
    user_details = {'Serial Number': ''}
    current_input = 'Serial Number'
    input_active = True

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if input_active:
                    if event.key == pygame.K_RETURN:
                        # Once the serial number is entered and RETURN is pressed, return user details
                        return user_details
                    elif event.key == pygame.K_BACKSPACE:
                        # Allow user to backspace if they make a mistake
                        user_details[current_input] = user_details[current_input][:-1]
                    else:
                        # Add the character to the serial number
                        user_details[current_input] += event.unicode

        screen.fill((0, 0, 0))  # Clear screen with black
        draw_text(screen, "Enter Participant Serial Number", (320, 50), center=True)
        draw_text(screen, "and then press Enter", (320, 80), center=True)
        draw_text(screen, f"Serial Number: {user_details['Serial Number']}", (320, 200), center=True)

        pygame.display.flip()
        pygame.time.Clock().tick(30)


def welcome_screen(screen):
    running = True
    while running:
        screen.fill((0, 0, 0))
        display_text(screen, "Welcome to the Experiment", (screen_width // 2, screen_height // 3))
        display_text(screen, "Press Enter to continue", (screen_width // 2, screen_height // 2))
        display_text(screen, "or Escape to exit", (screen_width // 2, screen_height // 1.7))

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
                pygame.quit()
                return False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                running = False

        pygame.display.flip()

    return True


def instruction_screen(screen, n_back_level, animal_sound):
    # Log event for the start of the N-BACK Instructions
    log_event(7, datetime.now().timestamp())
    screen.fill((0, 0, 0))  # Clear screen

    # Text settings
    font_size = 40  # Adjust font size for clarity and fit
    font = pygame.font.Font(None, font_size)
    color = (255, 255, 255)  # White color for text
    line_spacing = 10  # Spacing between lines

    # Instruction Text
    instructions = ["Navigate through the maze to find the exit.", ""]

    # Additional info for the N-back task explanation
    if n_back_level > 0:
        instructions.append("If a sound is the same as one you heard ")
        instructions.append(f"{n_back_level} steps ago,")
        if animal_sound:
            instructions.append("and!! it's an animal sound,")
            instructions.append("press SPACEBAR.")
        else:
            instructions.append("press SPACEBAR regardless of its type.")
    instructions.append(" ")
    instructions.append("Press SPACEBAR to start the level.")

    # Calculate the total height of the text block
    total_height = (font_size + line_spacing) * len(instructions) - line_spacing

    # Calculate the starting y position to vertically center the text
    start_y = (screen_height - total_height) // 2

    for i, text in enumerate(instructions):
        text_surface = font.render(text, True, color)
        rect = text_surface.get_rect(center=(screen_width // 2, start_y + i * (font_size + line_spacing)))
        screen.blit(text_surface, rect)

    pygame.display.flip()

    waiting_for_input = True
    while waiting_for_input:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                save_data_and_participant_info(experiment_data, user_details, stage_performance)
                return False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting_for_input = False

    return True


def create_maze_background(maze, cell_size):
    maze_background = pygame.Surface((maze.shape[1] * cell_size, maze.shape[0] * cell_size))
    maze_background.fill(BLACK)
    for y in range(maze.shape[0]):
        for x in range(maze.shape[1]):
            if maze[y, x] == 1:
                pygame.draw.rect(maze_background, WHITE, (x * cell_size, y * cell_size, cell_size, cell_size))
    return maze_background


def load_sounds(folder, sound_type):
    sounds = []
    for filename in os.listdir(folder):
        if filename.endswith('.wav'):  # Adjust the format as needed
            path = os.path.join(folder, filename)
            sound = pygame.mixer.Sound(path)
            sound_info = {'sound': sound, 'type': sound_type, 'filename': filename}
            sounds.append(sound_info)
    return sounds


def nasa_tlx_rating_screen():
    global stage_performance
    # Log event for the start of the NASA-TLX rating
    log_event(11, datetime.now().timestamp())

    # Define screen properties
    screen_width = 800
    screen_height = 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("NASA-TLX Rating")
    title_font = pygame.font.Font(None, 32)  # Larger font for the title
    # Define a new font size that is smaller to save space
    small_font_size = 22
    small_font = pygame.font.Font(None, small_font_size)

    # Define colors
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    BLACK = (0, 0, 0)
    GREEN = (0, 255, 0)

    # Categories and initial scale values
    categories = ["Mental Demand", "Performance", "Frustration"]
    scale_values = {category: 10 for category in categories}  # Mid-point of the scale

    # Title
    title = "NASA-TLX Rating"

    def draw_scales():
        # Draw title at the original position
        title_surface = title_font.render(title, True, WHITE)
        screen.blit(title_surface, (screen_width // 2 - title_surface.get_width() // 2, 10))

        # Calculate the vertical centering for the scales
        total_content_height = len(categories) * 80  # Assuming 80 pixels per category
        start_y_position = (screen_height - total_content_height) // 2

        # Center the scales horizontally
        for i, category in enumerate(categories):
            y_position = start_y_position + i * 80  # Spacing between questions
            text_label = small_font.render(category, True, WHITE)

            # Find the center of the text label relative to the screen width
            text_x_position = screen_width // 2 - text_label.get_width() // 2
            screen.blit(text_label, (text_x_position, y_position))

            # Draw the scale line centered
            line_start_x = screen_width // 2 - 250  # Line starting position adjusted for centering
            line_end_x = screen_width // 2 + 250  # Line ending position adjusted for centering
            pygame.draw.line(screen, WHITE, (line_start_x, y_position + 30), (line_end_x, y_position + 30), 2)

            # Calculate the position for the scale marker based on the current scale value
            marker_pos_x = line_start_x + (scale_values[category] * (500 / 19))
            pygame.draw.circle(screen, GREEN, (int(marker_pos_x), y_position + 30), 10)

            # Place "Very Low" and "Very High" labels at the ends of the scale line
            low_text = small_font.render("Very Low", True, WHITE)
            high_text = small_font.render("Very High", True, WHITE)

            # Draw the "Very Low" and "Very High" text aligned with the scale line ends
            screen.blit(low_text, (line_start_x, y_position + 50))
            screen.blit(high_text, (line_end_x - high_text.get_width(), y_position + 50))

    def handle_mouse_click(pos):

        # Calculate the vertical centering for the scales
        total_content_height = len(categories) * 80  # Assuming 80 pixels per category
        start_y_position = (screen_height - total_content_height) // 2

        for i, category in enumerate(categories):
            y_position = start_y_position + i * 80  # Ensure this matches the draw_scales logic
            line_start_x = screen_width // 2 - 250  # Line starting position adjusted for centering
            line_end_x = screen_width // 2 + 250  # Line ending position adjusted for centering

            # The clickable area for each scale
            clickable_area_start_y = y_position + 20  # Slightly above the line
            clickable_area_end_y = y_position + 40  # Slightly below the line

            if line_start_x <= pos[0] <= line_end_x and clickable_area_start_y <= pos[1] <= clickable_area_end_y:
                scale_values[category] = round((pos[0] - line_start_x) / (500 / 19))
                return

    # Position and draw the "Continue" button
    continue_button = pygame.Rect(screen_width // 2 - 50, screen_height - 80, 100, 40)  # Adjusted size and position

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if continue_button.collidepoint(event.pos):
                    running = False
                else:
                    handle_mouse_click(event.pos)

        screen.fill(BLACK)
        draw_scales()

        # Draw the "Continue" button
        pygame.draw.rect(screen, RED, continue_button)
        continue_text = small_font.render('Continue', True, WHITE)
        screen.blit(continue_text, (continue_button.x + 5, continue_button.y + 5))

        pygame.display.flip()

    # Add the scale values to the stage performance list as a dictionary
    for scale in categories:
        stage_performance[-1][scale] = scale_values[scale]
    # Log event for the end of the NASA-TLX rating
    log_event(12, datetime.now().timestamp())


def play_sound(sound_info):
    global sound_sequence
    pygame.mixer.Sound.play(sound_info['sound'])
    sound_sequence.append(sound_info['filename'])  # Use a unique identifier for the sound


def load_maze_from_file(dim):
    """
    Load a random maze from the pre-generated files based on dimension.
    """
    dir_path = f'ready_maze/dim{dim}'
    maze_files = [f for f in os.listdir(dir_path) if f.endswith('.json')]
    random_maze_file = random.choice(maze_files)
    maze_path = os.path.join(dir_path, random_maze_file)
    with open(maze_path, 'r') as file:
        maze_array = json.load(file)
    return np.array(maze_array)  # Convert back to a numpy array if needed


def setup_level(n_back_level, screen_width, screen_height):
    global experiment_ended, maze_size, maze_complexity, animal_sound, level_list, stage_performance

    if n_back_level == -1:
        n_back_level += 1
    else:
        # Analyze the latest performance from experiment_data
        # Placeholder for performance analysis logic
        errors = analyze_performance()
        stage_performance.append({'timestamp': datetime.now().timestamp(), 'error_rate': errors,
                                  'n_back_level': n_back_level, 'maze_size': maze_size,
                                  'animal_sound': animal_sound})
        print(f"Performance error: {errors}")

        # Criteria for performance evaluation
        acceptable_error_threshold = 0.3  # Example threshold
        significant_drop_threshold = 0.8  # Example threshold for significant performance drop

        # Logic to decide on increasing difficulty or ending the experiment
        if errors <= acceptable_error_threshold:
            # Decide which aspect to increase: This is a simplified example; implement your own logic
            if level_list[0] == 0:  # Increase the N-back level
                animal_sound = False
                n_back_level += 1
                level_list[0] = 1
            elif level_list[1] == 0:  # Increase the maze size
                maze_size += 5
                level_list[1] = 1
            elif level_list[2] == 0:  # Increase the maze complexity
                level_list[2] = 1
            else:
                animal_sound = True
                level_list = [0, 0, 0]  # Reset the level list
        elif errors > significant_drop_threshold:
            # send trigger to indicate that the subject has high error rates
            log_event(10, datetime.now().timestamp())

    # # Generate the new level's maze with updated parameters
    # maze = maze_to_ndarray(maze_size, 6, maze_complexity)

    # Adjusted code to load the maze instead of generating a new one
    maze = load_maze_from_file(maze_size)

    # Calculate the size of each cell to fit the screen
    maze_width = maze.shape[1]
    maze_height = maze.shape[0]
    cell_size = min(screen_width // maze_width, screen_height // maze_height)

    # Recalculate the screen dimensions to fit the maze
    screen_width = cell_size * maze_width
    screen_height = cell_size * maze_height

    # Create the maze background for the new level
    maze_background = create_maze_background(maze, cell_size)

    return maze, n_back_level, screen_width, screen_height, cell_size, maze_background


def calculate_performance_ratio(current_stat, prev_stat):
    if prev_stat > 0:
        return current_stat / prev_stat
    elif current_stat == 0:
        return 0
    else:
        return current_stat


def analyze_performance():
    global performance_ratios

    # Initialize counts of levels completed
    levels_completed = sum(data['trigger_id'] == 4 for data in experiment_data)

    if levels_completed == 1:  # For the first step, the error is 0
        return 0

    # Calculation of the amount of triggers of each type for the level that is now finished, and the previous level
    current_triggers = [0, 0, 0]

    for data in reversed(experiment_data):
        if data['trigger_id'] == 9:
            performance_ratios['end'].append(data['timestamp'])
        if data['trigger_id'] == 4:
            performance_ratios['start'].append(data['timestamp'])
            break
        if data['trigger_id'] < 4:
            current_triggers[data['trigger_id'] - 1] += 1

    performance_ratios['TP'].append(current_triggers[1] / (current_triggers[1] + current_triggers[2])
                                    if current_triggers[1] + current_triggers[2] > 0 else 1)

    if current_triggers[0] + current_triggers[1] > 0:
        performance_ratios['FP'].append(current_triggers[0] / (current_triggers[0] + current_triggers[1]))
    elif current_triggers[2] > 0:
        performance_ratios['FP'].append(1)
    else:
        performance_ratios['FP'].append(0)

    if levels_completed < 4:
        return 0

    # Calculate the rates of the duration, True Positives and False Positives
    opp_tp_current_rate = calculate_performance_ratio(1 - performance_ratios['TP'][-1],
                                                      1 - performance_ratios['TP'][-2])
    opp_tp_prev_rate = calculate_performance_ratio(1 - performance_ratios['TP'][-2], 1 - performance_ratios['TP'][-3])
    fp_current_rate = calculate_performance_ratio(performance_ratios['FP'][-1], performance_ratios['FP'][-2])
    fp_prev_rate = calculate_performance_ratio(performance_ratios['FP'][-2], performance_ratios['FP'][-3])

    duration_current = (performance_ratios['end'][-1] - performance_ratios['start'][-1])
    duration_prev = (performance_ratios['end'][-2] - performance_ratios['start'][-2])
    duration_prev_prev = (performance_ratios['end'][-3] - performance_ratios['start'][-3])

    print(
        f"duration_current: {duration_current}, duration_prev: {duration_prev}, duration_prev_prev: {duration_prev_prev}")

    # Calculate the rates
    duration_rate = (duration_current * duration_prev_prev) / duration_prev ** 2
    tp_rate = calculate_performance_ratio(opp_tp_current_rate, opp_tp_prev_rate)
    fp_rate = calculate_performance_ratio(fp_current_rate, fp_prev_rate)

    print(f"duration_rate: {duration_rate}, tp_rate: {tp_rate}, fp_rate: {fp_rate}")

    # Return the average of the rates
    return duration_rate / 9 + tp_rate * 4 / 9 + fp_rate * 4 / 9


def is_maze_completed(player_x, player_y, maze):  # check if the player reached the exit
    # Check if player reached the exit Assuming exit is at [-2, -1]
    if player_x == maze.shape[1] - 1 and player_y == maze.shape[0] - 2:
        return True
    else:
        return False


def save_data_and_participant_info(experiment_data, user_details, stage_performance):
    serial_number = user_details['Serial Number']
    folder_name = f"S{serial_number.zfill(3)}"
    os.makedirs(folder_name, exist_ok=True)

    # Save experiment data
    csv_file_path = os.path.join(folder_name, 'experiment_data.csv')
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['timestamp', 'trigger_id'])
        writer.writeheader()
        for data in experiment_data:
            writer.writerow(data)

    # Save participant details
    details_file_path = os.path.join(folder_name, 'participant_details.txt')
    with open(details_file_path, 'w') as file:
        for key, value in user_details.items():
            file.write(f"{key}: {value}\n")

    # Save error performance
    error_file_path = os.path.join(folder_name, 'stage_performance.csv')
    with open(error_file_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['timestamp', 'error_rate', 'n_back_level', 'maze_size',
                                                  'animal_sound', 'Mental Demand',
                                                  'Performance', 'Frustration'])
        writer.writeheader()
        for data in stage_performance:
            writer.writerow(data)


# Load sounds from both folders
object_sounds = load_sounds(os.path.join("back_sound", "sound_object"), "object")
animal_sounds = load_sounds(os.path.join("back_sound", "sound_animal"), "animal")

sound_sequence = []  # Reset for the new level

# Initialize Pygame
pygame.init()

# Timers and intervals
initial_delay = 1500  # 1.5 seconds before the first sound
sound_interval = 2000  # 2 seconds between sounds
initial_timer = 0
interval_timer = 0

# Screen dimensions (constant size)
screen_width = 600
screen_height = 600

experiment_ended = False  # New variable to control the experiment's end
experiment_start_time = datetime.now()  # Track the start time of the experiment

# Initialize the screen
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Maze experiment")

# Show welcome screen
if not welcome_screen(screen):
    exit()

# Define the screen size
screen_width = 640
screen_height = 480
screen = pygame.display.set_mode((screen_width, screen_height))

# Initialize pygame
pygame.display.set_caption("User Details Input")

user_details = input_screen()

# Maze parameters
dim = 30  # Size of the maze

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Calculate the size of each cell to fit the screen
maze_width = 2 * dim + 1
maze_height = 2 * dim + 1
cell_size = min(screen_width // maze_width, screen_height // maze_height)

# Adjust screen dimensions to fit the maze exactly
screen_width = cell_size * maze_width
screen_height = cell_size * maze_height
screen = pygame.display.set_mode((screen_width, screen_height))

# Player position - starting at the entrance of the maze
player_x, player_y = 0, 1  # Adjusted to start at the maze entrance

# Player speed (number of cells per frame)
speed = 1

# Key state
key_pressed = None

# Level parameters
initial_screen_width = 600  # Initial screen dimensions
initial_screen_height = 600
maze, n_back_level, screen_width, screen_height, cell_size, maze_background = \
    setup_level(-1, initial_screen_width, initial_screen_height)

instruction_screen_set = pygame.display.set_mode((initial_screen_width, initial_screen_height))

# Log event for the start of the experiment
log_event(6, datetime.now().timestamp())

# Call the instruction screen with the updated window size
if not instruction_screen(instruction_screen_set, n_back_level, animal_sound):
    exit()  # Exit if the user closes the window or presses ESCAPE

screen = pygame.display.set_mode((screen_width, screen_height))

# Create the maze background for the first level
maze_background = pygame.Surface((screen_width, screen_height))

# Draw the maze on the background surface
maze_background.fill(BLACK)
for y in range(maze.shape[0]):
    for x in range(maze.shape[1]):
        if maze[y, x] == 1:
            pygame.draw.rect(maze_background, WHITE, (x * cell_size, y * cell_size, cell_size, cell_size))

# track the correctness of the response
expected_response = False
response_made = False

# Maze loop
running = True

# Log event for the start of a new level
log_event(4, datetime.now().timestamp())

# Main game loop
while running:
    # Reset the response flag for the new interval
    response_made = False  # Reset response flag for the new interval

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False
        elif event.type == pygame.KEYDOWN:
            key_pressed = event.key
        elif event.type == pygame.KEYUP:
            if event.key == key_pressed:
                key_pressed = None

        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            response_made = True
            # Determine correctness and log event
            if expected_response:  # If a response was expected
                log_event(2, datetime.now().timestamp())  # Correct response, Trigger ID 2
            else:
                log_event(1, datetime.now().timestamp())  # Incorrect response, Trigger ID 1
            expected_response = False  # Reset for the next sound

    # Blit the maze background
    screen.blit(maze_background, (0, 0))

    # Move player based on key state
    if key_pressed == pygame.K_LEFT and player_x - speed >= 0 and maze[player_y, player_x - speed] == 0:
        player_x -= speed
    elif key_pressed == pygame.K_RIGHT and player_x + speed < maze.shape[1] and maze[player_y, player_x + speed] == 0:
        player_x += speed
    elif key_pressed == pygame.K_UP and player_y - speed >= 0 and maze[player_y - speed, player_x] == 0:
        player_y -= speed
    elif key_pressed == pygame.K_DOWN and player_y + speed < maze.shape[0] and maze[player_y + speed, player_x] == 0:
        player_y += speed

    # Sound playback logic
    if n_back_level > 0:
        delta_time = pygame.time.Clock().tick(30)
        if initial_timer < initial_delay:
            initial_timer += delta_time
        else:
            interval_timer += delta_time

        if initial_timer >= initial_delay and interval_timer >= sound_interval:

            # Check if the player's response is correct
            if expected_response and not response_made:
                log_event(3, datetime.now().timestamp())  # Log missed response
                expected_response = False  # Reset for the next interval

            # # Randomly choose a sound list and then a sound from that list
            # chosen_list = random.choice([object_sounds, animal_sounds])

            # Randomly choose a sound list and then a sound from that list
            chosen_list = random.choice([object_sounds, animal_sounds])
            chosen_sound_info = random.choice(chosen_list)  # This is a dictionary
            play_sound(chosen_sound_info)  # Updated to use play_sound
            # Determine if this sound requires a response based on N-BACK rule
            if (len(sound_sequence) >= n_back_level + 1 and
                    sound_sequence[-n_back_level - 1] == chosen_sound_info['filename']):
                if not animal_sound:
                    expected_response = True  # A response is expected for this sound
                elif chosen_sound_info['type'] == 'animal':
                    expected_response = True

            interval_timer = 0  # Reset the interval timer
            response_made = False  # Reset response flag for the new interval

    # Draw the player
    pygame.draw.rect(screen, RED, (player_x * cell_size, player_y * cell_size, cell_size, cell_size))

    # Check if the maze is completed
    if is_maze_completed(player_x, player_y, maze):
        # Log event for the end of the level
        log_event(9, datetime.now().timestamp())
        if (datetime.now() - experiment_start_time).total_seconds() > 3600:
            experiment_ended = True
        # set up the new level
        maze, n_back_level, screen_width, screen_height, cell_size, maze_background = \
            setup_level(n_back_level, screen_width, screen_height)

        # Call the rating screen function after a maze is completed
        nasa_tlx_ratings = nasa_tlx_rating_screen()
        if experiment_ended:
            # Log event for the end of the game
            log_event(5, datetime.now().timestamp())
            # Save data before exiting after completing all levels
            save_data_and_participant_info(experiment_data, user_details, stage_performance)
            break  # End the game after the last level

        # Reset the sound sequence for the new level
        sound_sequence = []

        # change the screen size back to instruction screen size
        instruction_screen_set = pygame.display.set_mode((initial_screen_width, initial_screen_height))
        # Call the instruction screen with the updated window size
        if not instruction_screen(instruction_screen_set, n_back_level, animal_sound):
            exit()  # Exit if the user closes the window or presses ESCAPE

        # Log event for the end of the N-BACK Instructions
        log_event(8, datetime.now().timestamp())

        # Log event for the start of a new level
        log_event(4, datetime.now().timestamp())

        # Adjust the screen dimensions to fit the new maze
        screen = pygame.display.set_mode((screen_width, screen_height))

        # Reset player position and other necessary variables for the new level
        player_x, player_y = 0, 1  # Reset player position to the start of the maze

    pygame.display.flip()
    pygame.time.Clock().tick(30)  # Limit to 30 frames per second

# Place outside the while loop, to handle game exit through closing the window or pressing escape.
save_data_and_participant_info(experiment_data, user_details, stage_performance)
pygame.quit()
