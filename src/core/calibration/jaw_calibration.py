import pygame
import random
import winsound
from datetime import datetime
from src.core.data_logging import log_event

def jaw_calibration(state, stim_flag):

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
    state.screen.fill((0, 0, 0))
    # Calculate the total height of the text block
    total_height = (font_size + line_spacing) * len(instruction_text) - line_spacing
    # Calculate the starting y position to vertically center the text
    start_y = (state.screen_height - total_height) // 2

    for i, text in enumerate(instruction_text):
        text_surface = font.render(text, True, color)
        rect = text_surface.get_rect(center=(state.screen_width // 2, start_y + i * (font_size + line_spacing)))
        state.screen.blit(text_surface, rect)

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
        state.screen.fill((0, 0, 0))
        total_height = (font_size + line_spacing) * len(prep_text) - line_spacing
        start_y = (state.screen_height - total_height) // 2

        for i, text in enumerate(prep_text):
            text_surface = font.render(text, True, color)
            rect = text_surface.get_rect(center=(state.screen_width // 2, start_y + i * (font_size + line_spacing)))
            state.screen.blit(text_surface, rect)

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

        state.screen.fill((0, 0, 0))
        total_height = (font_size + line_spacing) * len(perform_text) - line_spacing
        start_y = (state.screen_height - total_height) // 2

        for i, text in enumerate(perform_text):
            text_surface = font.render(text, True, color)
            rect = text_surface.get_rect(center=(state.screen_width // 2, start_y + i * (font_size + line_spacing)))
            state.screen.blit(text_surface, rect)

        pygame.display.flip()

        # Log the trigger for the movement
        trigger_id = movement_triggers[movement]
        log_event(trigger_id, datetime.now().timestamp(), state, stim_flag)

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
    state.screen.fill((0, 0, 0))
    total_height = (font_size + line_spacing) * len(end_text) - line_spacing
    start_y = (state.screen_height - total_height) // 2

    for i, text in enumerate(end_text):
        text_surface = font.render(text, True, color)
        rect = text_surface.get_rect(center=(state.screen_width // 2, start_y + i * (font_size + line_spacing)))
        state.screen.blit(text_surface, rect)

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