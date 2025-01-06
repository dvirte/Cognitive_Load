import pygame
import winsound
from datetime import datetime
from src.core.data_logging import log_event

def white_dot_calibration(state, stim_flag):
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
    state.screen.fill((0, 0, 0))
    # Draw the white dot at the center
    dot_radius = 10  # Adjust the size as needed
    pygame.draw.circle(state.screen, (255, 255, 255),
                       (state.screen_width // 2, state.screen_height // 2), dot_radius)
    pygame.display.flip()

    # Log the start of the white dot calibration
    log_event(21, datetime.now().timestamp(), state, stim_flag)

    # Wait for 12 seconds while handling events
    start_time = pygame.time.get_ticks()
    while pygame.time.get_ticks() - start_time < 7000:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        pygame.time.delay(100)

    # Play the beep sound to signal closing eyes
    winsound.Beep(2500, 500)
    # Log the beep event to close eyes
    log_event(22, datetime.now().timestamp(), state, stim_flag)

    # Continue displaying the dot
    pygame.draw.circle(state.screen, (255, 255, 255),
                       (state.screen_width // 2, state.screen_height // 2), dot_radius)
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
    log_event(23, datetime.now().timestamp(), state, stim_flag)

    # Clear the screen
    state.screen.fill((0, 0, 0))
    pygame.display.flip()