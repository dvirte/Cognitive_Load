import pygame
import winsound
from datetime import datetime
from src.core.calibration.white_dot_calibration import white_dot_calibration
from src.core.calibration.jaw_calibration import jaw_calibration
from src.core.data_logging import log_event
from src.core.utils import display_text

def calibration_screen(state, stim_flag):
    # Call the white dot calibration step
    white_dot_calibration(state, stim_flag)

    # Call the jaw calibration step
    jaw_calibration(state, stim_flag)

    # Define a function to display a slide with instructions
    def display_calibration_instruction(instruction_text, start_trigger_id):
        if start_trigger_id == 17:
            log_event(start_trigger_id, datetime.now().timestamp(), state, stim_flag)

        # Text settings
        font_size = 40  # Adjust font size for clarity and fit
        font = pygame.font.Font(None, font_size)
        color = (255, 255, 255)  # White color for text
        line_spacing = 10  # Spacing between lines

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

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        running = False
            pygame.display.flip()

        if start_trigger_id != 17:
            # Display the action slide
            state.screen.fill((0, 0, 0))  # Clear screen
            display_text(state, "Please wait for the beep.", font_size=40, y_offset=0)  # Centered text
            pygame.display.flip()
            pygame.time.wait(2500)  # Wait for 5 seconds

            # Perform the actions with beep sounds and triggers
            for _ in range(3):
                # Play a beep sound
                winsound.Beep(2500, 500)
                log_event(start_trigger_id, datetime.now().timestamp(), state, stim_flag)
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

    log_event(18, datetime.now().timestamp(), state, stim_flag)