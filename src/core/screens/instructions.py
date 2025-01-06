import pygame
import random
import winsound
from datetime import datetime
# from src.core.data_logging import log_event, save_data_and_participant_info

from src.core.utils import display_text

def instruction_screen(state, stim_flag, user_details):

    # Import the log_event and save_data_and_participant_info functions from the data_logging module
    from src.core.data_logging import log_event, save_data_and_participant_info

    # Log event for the start of the N-BACK Instructions
    log_event(7, datetime.now().timestamp(), state, stim_flag)
    state.screen.fill((0, 0, 0))  # Clear screen

    # Text settings
    font_size = 60  # Adjusted font size for readability
    font = pygame.font.Font(None, font_size)
    color = (255, 255, 255)  # White color for text
    line_spacing = 30  # Spacing between lines

    # Instruction Text
    instructions = ["Navigate through the maze to find the exit.", ""]

    # Additional info for the N-back task explanation
    if state.n_back_level > 0:
        instructions.append("If a sound is the same as one you heard ")
        instructions.append(f"{state.n_back_level} steps ago,")

        if state.animal_sound:
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
    start_y = (state.screen_height - total_height) // 2

    for i, text in enumerate(instructions):
        # Render each line of text
        text_surface = font.render(text, True, color)
        # Get the rectangle of the text and explicitly center it horizontally and position vertically
        rect = text_surface.get_rect()  # Get rect without centering yet
        rect.center = (state.screen_width // 2, start_y + i * (font_size + line_spacing))  # Center horizontally
        state.screen.blit(text_surface, rect)

    pygame.display.flip()

    waiting_for_input = True
    while waiting_for_input:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                save_data_and_participant_info(state, stim_flag, user_details)
                pygame.quit()
                return False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting_for_input = False


    return True

def end_experiment_screen(state):
    not_stop_game = True
    while not_stop_game:
        state.screen.fill((0, 0, 0))  # Clear the screen with black

        # Display the thank-you messages
        display_text(state, "Thank you very much for your participation.", font_size=60, y_offset=-50)
        display_text(state, "Please do not click on anything.", font_size=60, y_offset=50)

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