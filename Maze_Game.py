import pygame
import os
import random
import numpy as np
import csv
from datetime import datetime
import pylsl
import json
import wave
import winsound
import copy

# Check if the stimulation controller is available
try:
    from stimulation_modules.stimulation_controller.example_use_STG5controller import stim_controller
    stim_flag = True
except ImportError:
    print("Stimulation controller not available")
    stim_flag = False

# Initialize LSL Stream
outlet = pylsl.StreamOutlet(pylsl.StreamInfo("Trigger_Cog", "Markers", 1, 0, pylsl.cf_int32, "myuidw43536"))

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
#    10: 'Indicates when the subject has high error rates'
#    11: 'Indicates start of the NASA-TLX rating'
#    13: 'Indicates for calibration: start blinking'
#    14: 'Indicates for calibration: start jaw stiffness'
#    15: 'Indicates for calibration: start frowning'
#    16: 'Indicates for calibration: start yawning'
#    17: 'Indicates for calibration: start calibration'
#    18: 'Indicates for calibration: end calibration'
#    19: 'Indicates that the participant skipped the maze in the middle.'
#    20: 'Indicates that a new maze is started because the previous maze was completed in less than 1.5 minutes.'
#    21: 'Calibration step: start of white dot with eyes open',
#    22: 'Calibration step: beep to close eyes',
#    23: 'Calibration step: end of white dot calibration',
#    24: 'Jaw Calibration: Clenching',
#    25: 'Jaw Calibration: Jaw opening',
#    26: 'Jaw Calibration: Chewing',
#    27: 'Jaw Calibration: Resting',
# }


experiment_data = []  # Initialize an empty list to store event data
maze_size = 5  # Initial maze size
current_threshold = 0.3  # Initial error threshold
maze_complexity = 0.8  # Initial maze complexity
animal_sound = True  # The n-back task only takes into account if the sound played is from animals and not inanimate
level_list = [0, 0, 0]  # List to keep track of which aspect to increase next
performance_ratios = {'TP': [], 'FP': [], 'start': [], 'end': []}  # Initialize performance ratios
stage_performance = []  # Initialize list to store high error performance
path_of_maze = [] # Initialize list to store the path of the maze
baseline_maze = 0  # Play 10 levels of maze without any n-back task
amount_of_levels = 10  # Amount of levels to play before starting the N-back task
time_of_experiment = 45  # Time in minutes for the experiment

# Initialize sound delay
sound_delay = 500  # Initial delay between sounds in milliseconds (0.5 seconds)
min_sound_delay = 500  # Minimum delay (0.5 seconds)
max_sound_delay = 5000  # Maximum delay (5 seconds)


def log_event(trigger_id, timestamp):
    global experiment_data
    # Log event data to the variable
    experiment_data.append({'timestamp': timestamp, 'trigger_id': trigger_id})

    # Send the trigger to the LSL stream
    outlet.push_sample([trigger_id])
    print(f"Trigger {trigger_id}")
    if stim_flag:
        stim_controller.start_stimulation(bursts_count=1)


def display_text(screen, text, font_size=50, color=(255, 255, 255), y_offset=0):
    font = pygame.font.Font(None, font_size)
    text_surface = font.render(text, True, color)
    # Set the text position dynamically with a vertical offset
    rect = text_surface.get_rect(center=(screen_width // 2, (screen_height // 2) + y_offset))
    screen.blit(text_surface, rect)


def draw_text(surface, text, x, y, font_size=32, color=(255, 255, 255), center=False):
    font = pygame.font.Font(None, font_size)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()

    if center:
        # Center the text horizontally around the x and place it at the y position
        text_rect.center = (x, y)
    else:
        # Place text in the top left corner of the x, y
        text_rect.topleft = (x, y)

    surface.blit(text_surface, text_rect)


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

        # Calculate positions dynamically based on screen height
        title_y = screen_height // 3
        instruction_y = title_y + 50  # Adjust the spacing between lines
        input_y = instruction_y + 150

        # Draw text centered horizontally
        draw_text(screen, "Enter Participant Serial Number", screen_width // 2, title_y, center=True, font_size=50)
        draw_text(screen, "and then press Enter", screen_width // 2, instruction_y, center=True, font_size=50)
        draw_text(screen, f"Serial Number: {user_details['Serial Number']}", screen_width // 2, input_y, center=True, font_size=50)

        pygame.display.flip()
        pygame.time.Clock().tick(30)


def welcome_screen(screen):
    running = True
    while running:
        screen.fill((0, 0, 0))
        display_text(screen, "Welcome to the Experiment", font_size=50, y_offset=-100)  # Move this line up
        display_text(screen, "Press Enter to continue", font_size=40, y_offset=20)       # Centered
        display_text(screen, "or Escape to exit", font_size=40, y_offset=90)           # Move this line down

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
                pygame.quit()
                return False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                running = False

        pygame.display.flip()

    return True


def white_dot_calibration(screen):
    # Display the instruction page
    instruction_text = [
        "A white dot will appear on the screen.",
        "Please look at it without blinking.",
        "When the beep sounds, please close your eyes",
        "and keep looking at where the dot is until the next beep is heard.",
        "Press Enter to continue."
    ]

    # Text settings
    font_size = 60
    font = pygame.font.Font(None, font_size)
    color = (255, 255, 255)  # White color for text
    line_spacing = 10  # Spacing between lines

    # Clear the screen
    screen.fill((0, 0, 0))
    # Calculate the total height of the text block
    total_height = (font_size + line_spacing) * len(instruction_text) - line_spacing
    # Calculate the starting y position to vertically center the text
    start_y = (screen_height - total_height) // 2

    for i, text in enumerate(instruction_text):
        text_surface = font.render(text, True, color)
        rect = text_surface.get_rect(center=(screen_width // 2, start_y + i * (font_size + line_spacing)))
        screen.blit(text_surface, rect)

    pygame.display.flip()

    # Wait for the participant to press Enter
    waiting_for_input = True
    while waiting_for_input:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    waiting_for_input = False
            elif event.type == pygame.QUIT:
                pygame.quit()
                return

    # Start the calibration
    screen.fill((0, 0, 0))
    # Draw the white dot at the center
    dot_radius = 10  # Adjust the size as needed
    pygame.draw.circle(screen, (255, 255, 255), (screen_width // 2, screen_height // 2), dot_radius)
    pygame.display.flip()

    # Log the start of the white dot calibration
    log_event(21, datetime.now().timestamp())

    # Wait for 12 seconds while handling events
    start_time = pygame.time.get_ticks()
    while pygame.time.get_ticks() - start_time < 12000:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        pygame.time.delay(100)

    # Play the beep sound to signal closing eyes
    winsound.Beep(2500, 500)
    # Log the beep event to close eyes
    log_event(22, datetime.now().timestamp())

    # Continue displaying the dot
    pygame.draw.circle(screen, (255, 255, 255), (screen_width // 2, screen_height // 2), dot_radius)
    pygame.display.flip()

    # Wait for another 45 seconds
    start_time = pygame.time.get_ticks()
    while pygame.time.get_ticks() - start_time < 45000:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        pygame.time.delay(100)

    # Play the beep sound to signal the end
    winsound.Beep(2500, 500)
    # Log the end of the white dot calibration
    log_event(23, datetime.now().timestamp())

    # Clear the screen
    screen.fill((0, 0, 0))
    pygame.display.flip()


def jaw_calibration(screen):
    global screen_width, screen_height

    # Define movements and corresponding trigger codes
    movements = ['Clenching', 'Jaw opening', 'Chewing', 'Resting']
    movement_triggers = {
        'Clenching': 24,
        'Jaw opening': 25,
        'Chewing': 26,
        'Resting': 27
    }

    # Create a list of movements, each repeated 3 times (adjust as needed)
    movement_list = movements * 15  # Total of 60 trials
    random.shuffle(movement_list)   # Randomize the order

    # Text settings
    font_size = 60
    font = pygame.font.Font(None, font_size)
    color = (255, 255, 255)
    line_spacing = 10

    # Display initial instructions
    instruction_text = [
        "A movement will be displayed on the screen.",
        "Prepare for the movement when you see it.",
        "Perform the movement when the beep sounds.",
        "Press Enter to start the calibration."
    ]

    # Clear the screen
    screen.fill((0, 0, 0))
    # Calculate the total height of the text block
    total_height = (font_size + line_spacing) * len(instruction_text) - line_spacing
    # Calculate the starting y position to vertically center the text
    start_y = (screen_height - total_height) // 2

    for i, text in enumerate(instruction_text):
        text_surface = font.render(text, True, color)
        rect = text_surface.get_rect(center=(screen_width // 2, start_y + i * (font_size + line_spacing)))
        screen.blit(text_surface, rect)

    pygame.display.flip()

    # Wait for the participant to press Enter to start
    waiting_for_input = True
    while waiting_for_input:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    waiting_for_input = False
            elif event.type == pygame.QUIT:
                pygame.quit()
                return
        pygame.time.delay(100)

    # Start the jaw calibration
    for movement in movement_list:
        # Display preparation instructions
        prep_text = [
            f"Prepare to perform: {movement}"
        ]

        # Clear the screen
        screen.fill((0, 0, 0))
        total_height = (font_size + line_spacing) * len(prep_text) - line_spacing
        start_y = (screen_height - total_height) // 2

        for i, text in enumerate(prep_text):
            text_surface = font.render(text, True, color)
            rect = text_surface.get_rect(center=(screen_width // 2, start_y + i * (font_size + line_spacing)))
            screen.blit(text_surface, rect)

        pygame.display.flip()

        # Wait for 3 seconds for preparation
        start_time = pygame.time.get_ticks()
        while pygame.time.get_ticks() - start_time < 3000:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            pygame.time.delay(100)

        # Play beep sound to signal performance
        winsound.Beep(750, 500)

        # Display performance instructions
        perform_text = [f"Now perform: {movement}"]

        screen.fill((0, 0, 0))
        total_height = (font_size + line_spacing) * len(perform_text) - line_spacing
        start_y = (screen_height - total_height) // 2

        for i, text in enumerate(perform_text):
            text_surface = font.render(text, True, color)
            rect = text_surface.get_rect(center=(screen_width // 2, start_y + i * (font_size + line_spacing)))
            screen.blit(text_surface, rect)

        pygame.display.flip()

        # Log the trigger for the movement
        trigger_id = movement_triggers[movement]
        log_event(trigger_id, datetime.now().timestamp())

        # Wait for 4 seconds while the participant performs the movement
        start_time = pygame.time.get_ticks()
        while pygame.time.get_ticks() - start_time < 4000:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            pygame.time.delay(100)

    # End of jaw calibration
    end_text = [
        "Jaw calibration is complete.",
        "Press Enter to continue."
    ]

    # Clear the screen
    screen.fill((0, 0, 0))
    total_height = (font_size + line_spacing) * len(end_text) - line_spacing
    start_y = (screen_height - total_height) // 2

    for i, text in enumerate(end_text):
        text_surface = font.render(text, True, color)
        rect = text_surface.get_rect(center=(screen_width // 2, start_y + i * (font_size + line_spacing)))
        screen.blit(text_surface, rect)

    pygame.display.flip()

    # Wait for the participant to press Enter to proceed
    waiting_for_input = True
    while waiting_for_input:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    waiting_for_input = False
            elif event.type == pygame.QUIT:
                pygame.quit()
                return
        pygame.time.delay(100)


def calibration_screen(screen):
    # Call the white dot calibration step
    white_dot_calibration(screen)

    # Call the jaw calibration step
    jaw_calibration(screen)

    # Define a function to display a slide with instructions
    def display_calibration_instruction(instruction_text, start_trigger_id):
        if start_trigger_id == 17:
            log_event(start_trigger_id, datetime.now().timestamp())

        # Text settings
        font_size = 40  # Adjust font size for clarity and fit
        font = pygame.font.Font(None, font_size)
        color = (255, 255, 255)  # White color for text
        line_spacing = 10  # Spacing between lines

        # Clear the screen
        screen.fill((0, 0, 0))
        # Calculate the total height of the text block
        total_height = (font_size + line_spacing) * len(instruction_text) - line_spacing
        # Calculate the starting y position to vertically center the text
        start_y = (screen_height - total_height) // 2

        for i, text in enumerate(instruction_text):
            text_surface = font.render(text, True, color)
            rect = text_surface.get_rect(center=(screen_width // 2, start_y + i * (font_size + line_spacing)))
            screen.blit(text_surface, rect)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        running = False
            pygame.display.flip()

        if start_trigger_id != 17:
            # Display the action slide
            screen.fill((0, 0, 0))  # Clear screen
            display_text(screen, "Please wait for the beep.", font_size=40, y_offset=0)  # Centered text
            pygame.display.flip()
            pygame.time.wait(2500)  # Wait for 5 seconds

            # Perform the actions with beep sounds and triggers
            for _ in range(3):
                # Play a beep sound
                winsound.Beep(2500, 500)
                log_event(start_trigger_id, datetime.now().timestamp())
                pygame.time.wait(5000)  # Wait for 5 seconds

    # Display the initial calibration instruction
    display_calibration_instruction(
        ["Now we will perform a calibration.", "Please press Enter to continue"],
        17)

    # Calibration steps
    calibration_steps = [
        (["Every time a beep sound is heard,", "please blink.", "", "To start, press Enter."], 13),
        (["Every time a beep sound is heard,", " please stiffen your jaw.", "", "To start, press Enter."], 14),
        (["Every time a beep sound is heard,", " please frown.", "", "To start, press Enter."], 15),
        (["Every time a beep sound is heard,", " please yawn.", "", "To start, press Enter."], 16)]

    for instruction_text, start_trigger_id in calibration_steps:
        display_calibration_instruction(instruction_text, start_trigger_id)

    log_event(18, datetime.now().timestamp())


def instruction_screen(screen, n_back_level, animal_sound):
    # Log event for the start of the N-BACK Instructions
    log_event(7, datetime.now().timestamp())
    screen.fill((0, 0, 0))  # Clear screen

    # Text settings
    font_size = 60  # Adjusted font size for readability
    font = pygame.font.Font(None, font_size)
    color = (255, 255, 255)  # White color for text
    line_spacing = 30  # Spacing between lines

    # Instruction Text
    instructions = ["Navigate through the maze to find the exit.", ""]

    # Additional info for the N-back task explanation
    if n_back_level > 0:
        instructions.append("If a sound is the same as one you heard ")
        instructions.append(f"{n_back_level} steps ago,")

        if animal_sound:
            instructions.append("and it's an animal sound,")  # Add instruction for animal sounds
        else:
            # Add a beep to clarify this instruction
            winsound.Beep(2500, 500)  # Frequency: 2500Hz, Duration: 500ms
        instructions.append("press SPACEBAR.")
    instructions.append(" ")
    instructions.append("Press SPACEBAR to start the level.")

    # Calculate the total height of the text block
    total_height = (font_size + line_spacing) * len(instructions) - line_spacing

    # Calculate the starting y position to vertically center the text block
    start_y = (screen_height - total_height) // 2

    for i, text in enumerate(instructions):
        # Render each line of text
        text_surface = font.render(text, True, color)
        # Get the rectangle of the text and explicitly center it horizontally and position vertically
        rect = text_surface.get_rect()  # Get rect without centering yet
        rect.center = (screen_width // 2, start_y + i * (font_size + line_spacing))  # Center horizontally
        screen.blit(text_surface, rect)

    pygame.display.flip()

    waiting_for_input = True
    while waiting_for_input:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                save_data_and_participant_info(experiment_data, user_details, stage_performance)
                pygame.quit()
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


def get_sound_duration(file_path):
    with wave.open(file_path, 'rb') as wav_file:  # Open the WAV file in read-binary mode
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
    return int(duration * 1000)  # Convert to milliseconds


def load_sounds(folder, sound_type):
    sounds = []
    for filename in os.listdir(folder):
        if filename.endswith('.wav'):  # Adjust the format as needed
            path = os.path.join(folder, filename)
            sound = pygame.mixer.Sound(path)
            duration = get_sound_duration(path)
            sound_info = {'sound': sound, 'type': sound_type,
                          'filename': filename, 'duration': duration}
            sounds.append(sound_info)
    return sounds


def nasa_tlx_rating_screen():
    global stage_performance
    # Log event for the start of the NASA-TLX rating
    log_event(11, datetime.now().timestamp())

    # Define screen properties
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("NASA-TLX Rating")
    title_font_size = 48  # Increase the title font size for more prominence
    title_font = pygame.font.Font(None, title_font_size)
    small_font_size = 30  # Increase font size for the scales
    small_font = pygame.font.Font(None, small_font_size)

    # Define colors
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    BLACK = (0, 0, 0)
    GREEN = (0, 255, 0)

    # Categories and initial scale values
    categories = ["Mental Demand", "Effort", "Performance", "Frustration"]
    scale_values = {category: 10 for category in categories}  # Mid-point of the scale

    # Title
    title = "NASA-TLX Rating"

    def draw_scales():
        # Draw title with more space from the top
        title_surface = title_font.render(title, True, WHITE)
        screen.blit(title_surface, (screen_width // 2 - title_surface.get_width() // 2, 50))  # 50px padding from top

        # Calculate the vertical centering for the scales, increase space between scales
        total_content_height = len(categories) * 120  # Increase spacing between categories
        start_y_position = (screen_height - total_content_height) // 2 + 50  # Add margin for title

        # Draw scales and labels
        for i, category in enumerate(categories):
            y_position = start_y_position + i * 120  # Adjusted spacing between questions
            text_label = small_font.render(category, True, WHITE)

            # Center the text label relative to the screen width
            text_x_position = screen_width // 2 - text_label.get_width() // 2
            screen.blit(text_label, (text_x_position, y_position))

            # Draw the scale line centered
            line_start_x = screen_width // 2 - 350  # Increase line length for the scale
            line_end_x = screen_width // 2 + 350
            pygame.draw.line(screen, WHITE, (line_start_x, y_position + 40), (line_end_x, y_position + 40), 4)  # Thicker line

            # Calculate the position for the scale marker based on the current scale value
            marker_pos_x = line_start_x + (scale_values[category] * (700 / 19))  # Adjust for longer line
            pygame.draw.circle(screen, GREEN, (int(marker_pos_x), y_position + 40), 15)  # Larger marker

            # Place "Very Low" and "Very High" labels at the ends of the scale line
            low_text = small_font.render("Very Low", True, WHITE)
            high_text = small_font.render("Very High", True, WHITE)

            # Draw the "Very Low" and "Very High" text aligned with the scale line ends
            screen.blit(low_text, (line_start_x, y_position + 60))
            screen.blit(high_text, (line_end_x - high_text.get_width(), y_position + 60))

    def handle_mouse_click(pos):
        total_content_height = len(categories) * 120  # Adjusted for new spacing
        start_y_position = (screen_height - total_content_height) // 2 + 50

        for i, category in enumerate(categories):
            y_position = start_y_position + i * 120
            line_start_x = screen_width // 2 - 350
            line_end_x = screen_width // 2 + 350

            # The clickable area for each scale
            clickable_area_start_y = y_position + 30  # Adjust based on new scale position
            clickable_area_end_y = y_position + 50

            if line_start_x <= pos[0] <= line_end_x and clickable_area_start_y <= pos[1] <= clickable_area_end_y:
                scale_values[category] = round((pos[0] - line_start_x) / (700 / 19))  # Adjust for new scale width
                return

    # Position and draw the "Continue" button
    continue_button = pygame.Rect(screen_width // 2 - 50, screen_height - 120, 120, 50)  # Move button up a bit

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
        screen.blit(continue_text, (continue_button.x + 10, continue_button.y + 10))

        pygame.display.flip()

    # Save scale values after completing the rating
    for scale in categories:
        stage_performance[-1][scale] = scale_values[scale]


def calculate_nasa_weight():
    # Set the display to full screen
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    screen_width, screen_height = screen.get_size()  # Update dimensions to the full screen size
    pygame.display.set_caption("NASA-TLX Weight Calculation")

    categories = ["Mental Demand", "Performance", "Effort", "Frustration"]
    comparisons = [(i, j) for i in categories for j in categories if i < j]
    results = {category: 0 for category in categories}
    # Shuffle the comparisons to avoid order bias
    random.shuffle(comparisons)

    # Larger font size for better readability in full screen
    font = pygame.font.Font(None, 60)  # Increased font size
    question_font = pygame.font.Font(None, 80)  # Larger font for the question

    WHITE = (255, 255, 255)

    def draw_option(text, position):
        text_surface = font.render(text, True, WHITE)
        rect = text_surface.get_rect(center=position)
        screen.blit(text_surface, rect)
        return rect

    running = True
    for comparison in comparisons:
        if not running:
            break

        screen.fill((0, 0, 0))
        question = "Which had a greater impact on your workload?"
        # Center the question at the top part of the screen
        question_surface = question_font.render(question, True, WHITE)
        question_rect = question_surface.get_rect(center=(screen_width // 2, screen_height // 4))
        screen.blit(question_surface, question_rect)

        comparison = [comparison[0], comparison[1]]
        random.shuffle(comparison)

        # Center the options on the screen
        left_option = draw_option(comparison[0], (screen_width // 3, screen_height // 2))
        right_option = draw_option(comparison[1], (2 * screen_width // 3, screen_height // 2))

        pygame.display.flip()

        choosing = True
        while choosing:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    choosing = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos
                    if left_option.collidepoint(mouse_pos):
                        results[comparison[0]] += 1
                        choosing = False
                    elif right_option.collidepoint(mouse_pos):
                        results[comparison[1]] += 1
                        choosing = False

    return results


def play_sound(sound_info):
    global sound_sequence
    pygame.mixer.Sound.play(sound_info['sound'])
    sound_sequence.append(sound_info)  # Use a unique identifier for the sound
    return sound_info['duration']


def load_maze_from_file(dim):
    """
    Load a random maze from the pre-generated files based on dimension.
    """
    dir_path = f'ready_maze/dim{dim}'
    maze_files = [f for f in os.listdir(dir_path) if f.endswith('.json')]
    random_maze_file = random.choice(maze_files)
    maze_path = os.path.join(dir_path, random_maze_file)
    path_of_maze.append(maze_path)
    with open(maze_path, 'r') as file:
        maze_array = json.load(file)
    return np.array(maze_array)  # Convert back to a numpy array if needed


def setup_level(n_back_level, screen_width, screen_height, adjust_levels=True):
    global maze_size, maze_complexity, animal_sound, \
        level_list, stage_performance, baseline_maze, path_of_maze, amount_of_levels, sound_delay

    if adjust_levels:

        if baseline_maze < amount_of_levels + 2:
            baseline_maze += 1
            # Structured increase in difficulty for the training phase
            if baseline_maze <= 4:
                maze_size = 5  # First 4 rounds with maze size 5
            elif 5 <= baseline_maze <= 7:
                maze_size = 10  # Next 3 rounds with maze size 10
            elif 8 <= baseline_maze <= 10:
                maze_size = 15  # Last 3 rounds with maze size 15
            else:
                maze_size = 5  # Default to 5 for rounds beyond the specified training phase
        if baseline_maze > 1:
            # Analyze the latest performance from experiment_data
            # Placeholder for performance analysis logic
            error_rate, adjusted_threshold = analyze_performance()
            stage_performance.append({'timestamp': datetime.now().timestamp(), 'error_rate': error_rate,
                                      'n_back_level': n_back_level, 'maze_size': maze_size,
                                      'animal_sound': animal_sound, "path_of_maze": copy.deepcopy(path_of_maze),
                                      'sound_delay': sound_delay})
            path_of_maze = []
            print(f"Performance error: {error_rate}")

            # Logic to decide on increasing difficulty or ending the experiment
            if error_rate <= adjusted_threshold and baseline_maze == amount_of_levels+2:
                # Decide which aspect to increase: This is a simplified example; implement your own logic
                if level_list[0] == 0:  # Increase the N-back level
                    animal_sound = True
                    n_back_level += 1
                    level_list[0] = 1
                elif level_list[1] == 0:  # Increase the maze size
                    maze_size = min(maze_size + 5, 50)
                    level_list[1] = 1
                elif level_list[2] == 0:  # Increase the maze complexity
                    level_list[2] = 1
                else:
                    animal_sound = False
                    level_list = [0, 0, 0]  # Reset the level list

                if error_rate < 0.8 * adjusted_threshold:
                    # Decrease the delay between sounds by 0.5 seconds, down to the minimum
                    sound_delay = max(sound_delay - 500, min_sound_delay)
                    print("sound delay decreased")

            elif error_rate > adjusted_threshold:
                # send trigger to indicate that the subject has high error rates
                log_event(10, datetime.now().timestamp())
                # Increase the delay between sounds by 0.5 seconds, up to the maximum
                sound_delay = min(sound_delay + 500, max_sound_delay)
                print("sound delay increased")

    # Adjusted code to load the maze instead of generating a new one
    maze = load_maze_from_file(maze_size)

    # Calculate the size of each cell to fit the screen
    maze_width = maze.shape[1]
    maze_height = maze.shape[0]
    cell_size = min(screen_width // maze_width, screen_height // maze_height)

    # Calculate the offsets to center the maze
    offset_x = (screen_width - maze_width * cell_size) // 2
    offset_y = (screen_height - maze_height * cell_size) // 2

    # Create the maze background for the new level
    maze_background = create_maze_background(maze, cell_size)

    return maze, n_back_level, screen_width, screen_height, cell_size, maze_background, offset_x, offset_y


def calculate_performance_ratio(current_stat, prev_stat):
    if prev_stat > 0:
        return current_stat / prev_stat
    elif current_stat == 0:
        return 0
    else:
        return current_stat


def calculate_gradient(error_rates):
    """
    Calculate the performance gradient using a simple linear regression over
    the error rates of the last few rounds.
    """
    # Assuming error_rates is a list of floats
    # This is a placeholder for the actual calculation
    slope = np.polyfit(range(len(error_rates)), error_rates, 1)[0]
    return slope


def adjust_threshold(current_threshold, gradient, maze_size, base_maze_size=5, adjustment_factor=0.05):
    """
    Adjust the error threshold dynamically based on the performance gradient,
    current threshold, and maze size.
    """
    # Adjust the threshold based on maze size
    maze_size_factor = maze_size / base_maze_size
    maze_adjustment = (maze_size_factor - 1) * adjustment_factor

    # Dynamic adjustment based on the performance gradient
    if gradient < 0:  # Improvement shown
        new_threshold = current_threshold - adjustment_factor
    else:  # No improvement or worsening
        new_threshold = current_threshold + adjustment_factor

    # Ensure the new threshold doesn't become too lenient or too stringent
    new_threshold = min(max(new_threshold + maze_adjustment, 0.1), 1.0)  # Assuming thresholds should be between 0.1 and 1.0

    return new_threshold


def analyze_performance():
    global performance_ratios, stage_performance, current_threshold

    # Initialize counts of levels completed
    levels_completed = sum(data['trigger_id'] == 4 for data in experiment_data)

    if levels_completed == 1:  # For the first step, the error is 0
        return 0, 1

    # Calculation of the amount of triggers of each type for the level that is now finished, and the previous level
    current_triggers = [0, 0, 0]

    performance_ratios['end'].append(datetime.now().timestamp())

    for data in reversed(experiment_data):
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
        return 0, 1

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

    # Calculate the error rate
    error_rate = duration_rate / 9 + tp_rate * 4 / 9 + fp_rate * 4 / 9

    if len(stage_performance) < 10:
        adjusted_threshold = 1
    else:
        # List of last 5 error rates
        recent_error_rates = []
        for i in range(5):
            recent_error_rates.append(stage_performance[-5+i]['error_rate'])
        # Calculate the performance gradient for the last 5 error rates
        gradient = calculate_gradient(recent_error_rates)

        # Adjust the base error threshold based on the gradient
        # Assuming 'acceptable_error_threshold' is a predefined constant
        adjusted_threshold = adjust_threshold(current_threshold, gradient, maze_size)
        # Update the current threshold for the next level
        current_threshold = adjusted_threshold

    return error_rate, adjusted_threshold


def is_maze_completed(player_x, player_y, maze):  # check if the player reached the exit
    # Check if player reached the exit Assuming exit is at [-2, -1]
    if player_x == maze.shape[1] - 1 and player_y == maze.shape[0] - 2:
        return True
    else:
        return False


def complete_maze():
    global maze, n_back_level, screen_width, screen_height, cell_size, maze_background, offset_x, offset_y
    global key_pressed, sound_sequence, player_x, player_y, running, screen, stage_start_time

    # Reset the key state to avoid unintended movement
    key_pressed = None  # Reset key state

    # Calculate the elapsed time for the current maze
    elapsed_time = (datetime.now() - stage_start_time).total_seconds()

    if (datetime.now() - experiment_start_time).total_seconds() > time_of_experiment*60: # experiment ended
        # Call the rating screen function after a maze is completed
        error_rate, _ = analyze_performance()
        stage_performance.append({
            'timestamp': datetime.now().timestamp(),
            'error_rate': error_rate,  # Ensure this is updated
            'n_back_level': n_back_level,
            'maze_size': maze_size,
            'animal_sound': animal_sound,
            "path_of_maze": copy.deepcopy(path_of_maze),
            'sound_delay': sound_delay,
        })
        nasa_tlx_ratings = nasa_tlx_rating_screen()
        # experiment ended
        running = False  # Set running to False to exit the game loop
        return  # End the function

    elif elapsed_time < 60 and baseline_maze >= amount_of_levels + 2:  # Less than 1 minute
        # Log event indicating a new maze is started due to quick completion
        log_event(20, datetime.now().timestamp())  # Trigger ID 20

        # Set up a new maze with the same parameters without changing levels or showing rating screen
        maze, n_back_level, screen_width, screen_height, cell_size, maze_background, offset_x, offset_y = \
            setup_level(n_back_level, screen_width, screen_height, adjust_levels=False)

    else:
        # Set up the new level
        maze, n_back_level, screen_width, screen_height, cell_size, maze_background, offset_x, offset_y = \
            setup_level(n_back_level, screen_width, screen_height, adjust_levels=True)

        # Call the rating screen function after a maze is completed
        nasa_tlx_ratings = nasa_tlx_rating_screen()

        # Reset the sound sequence for the new level
        sound_sequence = []

        # Change the screen size back to instruction screen size
        instruction_screen_set = pygame.display.set_mode((screen_width, screen_height))

        # Call the instruction screen with the updated window size
        if not instruction_screen(instruction_screen_set, n_back_level, animal_sound):
            exit()  # Exit if the user closes the window or presses ESCAPE


        # Record the start time of the new stage
        stage_start_time = datetime.now()  # Add this line

    # Log event for the start of a new level
    if experiment_data[-1]['trigger_id'] != 20: # If the last trigger was not for a new maze in the same level
        log_event(4, datetime.now().timestamp())

    # Adjust the    screen dimensions to fit the new maze
    screen = pygame.display.set_mode((screen_width, screen_height))

    # Reset player position and other necessary variables for the new level
    player_x, player_y = 0, 1  # Reset player position to the start of the maze


def end_experiment_screen(screen):
    not_stop_game = True
    while not_stop_game:
        screen.fill((0, 0, 0))  # Clear the screen with black

        # Display the thank-you messages
        display_text(screen, "Thank you very much for your participation.", font_size=60, y_offset=-50)
        display_text(screen, "Please do not click on anything.", font_size=60, y_offset=50)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                not_stop_game = False
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    not_stop_game = False

        pygame.time.delay(100)  # Add a small delay to prevent high CPU usage


def save_data_and_participant_info(experiment_data, user_details, stage_performance):
    weights = calculate_nasa_weight()
    serial_number = user_details['Serial Number']
    folder_name = f"S{serial_number.zfill(3)}"
    os.makedirs(folder_name, exist_ok=True)

    # Log event for the end of the game
    log_event(5, datetime.now().timestamp())

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

    # Save error performance and NASA-TLX weights together
    error_file_path = os.path.join(folder_name, 'stage_performance.csv')
    with open(error_file_path, 'w', newline='') as file:
        # Include weight keys in the fieldnames
        fieldnames = list(stage_performance[0].keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        # Assuming weights should be included in every line of stage_performance data
        for data in stage_performance:
            writer.writerow(data)

    # Save NASA-TLX weights
    weights_file_path = os.path.join(folder_name, 'nasa_tlx_weights.csv')
    with open(weights_file_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=weights.keys())
        writer.writeheader()
        writer.writerow(weights)

    # End an experiment definitively
    end_experiment_screen(screen)


# Load sounds from both folders
object_sounds = load_sounds(os.path.join("back_sound", "sound_object"), "object")
animal_sounds = load_sounds(os.path.join("back_sound", "sound_animal"), "animal")

sound_sequence = []  # Reset for the new level

# Timers and intervals
time_since_last_sound = 0  # Time elapsed since the last sound played
sound_duration = 0  # Duration of the currently playing sound
sound_end_time = 0  # The calculated end time for the current sound
sound_to_play_next = True  # Flag to control when the next sound can be played

# Initialize the screen
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
screen_width, screen_height = screen.get_size()  # Update dimensions to the full screen size
pygame.display.set_caption("Maze experiment")

# Show welcome screen
if not welcome_screen(screen):
    exit()

# Initialize pygame
pygame.display.set_caption("User Details Input")

# Get user details
while True:
    user_details = input_screen()
    # check if user_details contain only numbers and its 3
    if user_details is not None and user_details['Serial Number'].isdigit() and len(user_details['Serial Number']) == 3:
        break

# Log event for the start of the experiment
log_event(6, datetime.now().timestamp())

# Perform the calibration
calibration_screen(screen)

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

# Player position - starting at the entrance of the maze
player_x, player_y = 0, 1  # Adjusted to start at the maze entrance

# Player speed (number of cells per frame)
speed = 1

# Key state
key_pressed = None

experiment_start_time = datetime.now()  # Track the start time of the experiment

# Level parameters
maze, n_back_level, screen_width, screen_height, cell_size, maze_background, offset_x, offset_y = \
    setup_level(0, screen_width, screen_height)

# Call the instruction screen with the updated window size
if not instruction_screen(screen, n_back_level, animal_sound):
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

#
cooldown_duration = 50  # milliseconds
last_move_time = 0  # track the last move time

# Log event for the start of a new level
log_event(4, datetime.now().timestamp())

# Record the start time of the first stage
stage_start_time = datetime.now()

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
            if event.key == pygame.K_e:     # Handle 'E' key press to skip the stage
                # Check if at least 5 minutes have passed since the stage started
                elapsed_time = (datetime.now() - stage_start_time).total_seconds()
                if elapsed_time >= 300:  # 300 seconds = 5 minutes
                    # Send unique trigger to signify that the user has stopped the maze in the middle.
                    log_event(19, datetime.now().timestamp())  # Trigger ID 19
                    # Simulate the maze completion
                    complete_maze()
                    pygame.display.flip()
                    pygame.time.Clock().tick(30)  # Limit to 30 frames per second
                    continue  # Go to next iteration of the loop
                else:
                    # Optionally provide feedback to the participant
                    print("You cannot skip the maze before 5 minutes have passed.")
        elif event.type == pygame.KEYUP:
            if event.key == key_pressed:
                key_pressed = None

        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            response_made = True
            # Determine correctness and log event
            if expected_response:  # If a response was expected
                log_event(2, datetime.now().timestamp())  # Correct response, Trigger ID 2
                print("correct response")
            else:
                log_event(1, datetime.now().timestamp())  # Incorrect response, Trigger ID 1
                print("incorrect response")
            expected_response = False  # Reset for the next sound

    # Blit the maze background
    screen.blit(maze_background, (offset_x, offset_y))

    current_time = pygame.time.get_ticks()
    if key_pressed and current_time - last_move_time > cooldown_duration:
        # Move player based on key state
        if key_pressed == pygame.K_LEFT and player_x - speed >= 0 and maze[player_y, player_x - speed] == 0:
            player_x -= speed
        elif key_pressed == pygame.K_RIGHT and player_x + speed < maze.shape[1] and maze[player_y, player_x + speed] == 0:
            player_x += speed
        elif key_pressed == pygame.K_UP and player_y - speed >= 0 and maze[player_y - speed, player_x] == 0:
            player_y -= speed
        elif key_pressed == pygame.K_DOWN and player_y + speed < maze.shape[0] and maze[player_y + speed, player_x] == 0:
            player_y += speed
        #
        last_move_time = current_time

    # Sound playback logic
    if n_back_level > 0:
        current_time = pygame.time.get_ticks()
        if current_time >= sound_end_time:

            # Check if the player's missed the response
            if expected_response and not response_made:
                log_event(3, datetime.now().timestamp())  # Log missed response
                print("missed response")
                expected_response = False  # Reset for the next interval

            # Play the next sound
            if random.randint(1, 100) <= 40 and len(sound_sequence) >= n_back_level + 1:
                chosen_sound_info = sound_sequence[-n_back_level]
            else:
                chosen_list = random.choice([object_sounds, animal_sounds])
                chosen_sound_info = random.choice(chosen_list)
            sound_duration = play_sound(chosen_sound_info)

            # Set the time when the next sound should be played
            sound_end_time = current_time + sound_duration + sound_delay

            # Determine if this sound requires a response based on N-BACK rule
            if (len(sound_sequence) >= n_back_level + 1 and
                    sound_sequence[-n_back_level - 1]['filename'] == chosen_sound_info['filename']):
                if not animal_sound:
                    expected_response = True  # A response is expected for this sound
                elif chosen_sound_info['type'] == 'animal':
                    expected_response = True

            response_made = False  # Reset response flag for the new interval

    # Draw the player
    pygame.draw.rect(screen, RED, ((player_x * cell_size + offset_x),
                                   (player_y * cell_size + offset_y), cell_size, cell_size))

    # Check if the maze is completed
    if is_maze_completed(player_x, player_y, maze):
        complete_maze()

    pygame.display.flip()
    pygame.time.Clock().tick(30)  # Limit to 30 frames per second

# Place outside the while loop, to handle game exit through closing the window or pressing escape.
save_data_and_participant_info(experiment_data, user_details, stage_performance)
pygame.quit()
