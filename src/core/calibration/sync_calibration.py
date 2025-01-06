import pygame
import winsound
from datetime import datetime
from src.core.utils import display_text
from src.core.data_logging import log_event

def synchronization_calibration(state, stim_flag):
    """
    Perform synchronization calibration.
    :param screen: Pygame screen for display.
    :param trigger_start: Trigger ID for starting calibration.
    :param trigger_end: Trigger ID for ending calibration.
    """
    # Display instructions
    instructions = [
        "Close your eyes and relax.",
        "Each time you hear a sound, close your eyes tightly.",
        "When you hear a double sound, open your eyes.",
        "",
        "Press Enter to continue."
    ]

    state.screen.fill((0, 0, 0))
    for i, line in enumerate(instructions):
        display_text(state, line, font_size=65, y_offset=-150 + i * 50)

    pygame.display.flip()

    # Wait for user to press Enter
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                waiting = False
            elif event.type == pygame.QUIT:
                pygame.quit()
                return

    state.screen.fill((0, 0, 0))
    display_text(state, "Close your eyes", font_size=60, y_offset=0)
    pygame.display.flip()

    # Perform calibration sequence
    sequence_durations = [7000, 5000, 5000, 5000]  # Durations in milliseconds

    # Log the start of the synchronization calibration
    log_event(28, datetime.now().timestamp(), state, stim_flag)

    for i, duration in enumerate(sequence_durations):
        pygame.time.wait(duration)

        if i < 3:  # Log single sound triggers
            # Log single sound trigger
            log_event(29, datetime.now().timestamp(), state, stim_flag)
            winsound.Beep(1000, 500)  # Single sound
        elif i == 3:  # Log double sound trigger
            # Log the end of the synchronization calibration
            log_event(30, datetime.now().timestamp(), state, stim_flag)
            winsound.Beep(1000, 300)  # Double sound
            pygame.time.wait(150)
            winsound.Beep(1000, 300)

    # Clear the screen after calibration
    state.screen.fill((0, 0, 0))
    pygame.display.flip()