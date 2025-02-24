import pygame
import random
import time
import pylsl
from data_logging import log_event
from nback_task import load_sounds, play_sound

def practice_n_back(state, cfg, object_sounds, animal_sounds):
    """Runs the N-Back practice phase before the main experiment."""

    def display_instruction(text_lines):
        """Displays multi-line instructions until the subject presses Enter."""
        state.screen.fill(cfg.BLACK)
        font = pygame.font.Font(None, 50)
        y_offset = state.screen_height // 3

        for i, line in enumerate(text_lines):
            text_surface = font.render(line, True, cfg.WHITE)
            text_rect = text_surface.get_rect(center=(state.screen_width // 2, y_offset + i * 50))
            state.screen.blit(text_surface, text_rect)

        pygame.display.flip()

        # Wait for Enter key
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    waiting = False

    def run_nback_task(object_sounds, animal_sounds, n_back_level, require_animal=False):
        """Runs the N-Back task until the subject gets 5 consecutive correct answers."""
        sound_sequence = []
        correct_streak = 0
        expected_response = False

        # Instruction text
        if require_animal:
            instructions = [
                "You will immediately hear different sounds.",
                "",
                "When the sound you hear is the same as the sound you heard before",
                "AND the sound is an animal, report it using the space bar.",
                "",
                "Press Enter to start."
            ]
            trigger_id = 33  # Trigger for second training phase
        else:
            instructions = [
                "You will immediately hear different sounds.",
                "",
                "When the sound you hear is the same as the sound you heard before,",
                "report it using the space bar.",
                "",
                "Press Enter to start."
            ]
            trigger_id = 31  # Trigger for first training phase

        display_instruction(instructions)
        log_event(trigger_id, time.time(), state, cfg.stim_flag)

        # Black screen for 3 seconds before starting
        state.screen.fill(cfg.BLACK)
        pygame.display.flip()
        pygame.time.wait(3000)

        while correct_streak < 3:
            state.screen.fill(cfg.BLACK)
            pygame.display.flip()
            pygame.time.wait(500)  # Short pause between trials

            # Select a sound
            if random.random() < 0.5 and len(sound_sequence) >= n_back_level:
                chosen_sound_info = sound_sequence[-n_back_level]
            else:
                chosen_list = random.choice([object_sounds, animal_sounds])
                chosen_sound_info = random.choice(chosen_list)

            sound_duration = play_sound(chosen_sound_info, state)

            # Determine expected response
            if len(sound_sequence) >= n_back_level:
                if sound_sequence[-n_back_level]['filename'] == chosen_sound_info['filename']:
                    if not require_animal or chosen_sound_info['type'] == 'animal':
                        expected_response = True

            # Wait for response
            response_made = False
            response_correct = False
            start_time = pygame.time.get_ticks()

            while pygame.time.get_ticks() - start_time < sound_duration + 2000:  # 2s response window
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        response_made = True
                        if expected_response:
                            response_correct = True
                            log_event(32, time.time(), state, cfg.stim_flag)  # Correct response
                        else:
                            log_event(30, time.time(), state, cfg.stim_flag)  # Incorrect response

            # Determine the feedback state
            feedback_text = ""
            feedback_color = None  # Default to None, ensuring no errors if no response is needed

            if response_made:
                if expected_response:
                    response_correct = True
                    correct_streak += 1
                    feedback_text = "V" # Checkmark symbol
                    feedback_color = (0, 255, 0)  # Green
                    log_event(32, time.time(), state, cfg.stim_flag)  # Correct response
                else:
                    correct_streak = 0
                    feedback_text = "X"
                    feedback_color = (255, 0, 0)  # Red
                    log_event(30, time.time(), state, cfg.stim_flag)  # Incorrect response (false positive)

            elif expected_response:
                correct_streak = 0
                feedback_text = "X"
                feedback_color = (255, 0, 0)  # Red
                log_event(34, time.time(), state, cfg.stim_flag)  # Missed expected response

            # Display feedback if there is something to show
            if feedback_text:
                state.screen.fill(cfg.BLACK)
                font = pygame.font.Font(None, 100)
                feedback_surface = font.render(feedback_text, True, feedback_color)
                feedback_rect = feedback_surface.get_rect(center=(state.screen_width // 2, state.screen_height // 2))
                state.screen.blit(feedback_surface, feedback_rect)
                pygame.display.flip()
                pygame.time.wait(2000)  # Show feedback for 2 seconds

            # Store sound sequence
            sound_sequence.append(chosen_sound_info)
            expected_response = False  # Reset for the next trial
            print(sound_sequence)


        # Training phase complete message
        display_instruction(["Great job!", "You have completed the practice.", "Press Enter to continue."])

    # Run first practice phase (Basic N-BACK)
    run_nback_task(object_sounds, animal_sounds, n_back_level=1, require_animal=False, )

    # Run second practice phase (N-BACK with animal filtering)
    run_nback_task(object_sounds, animal_sounds, n_back_level=1, require_animal=True)
