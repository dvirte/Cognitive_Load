import pygame
import pylsl
import os
import random
import config as cfg
from datetime import datetime
from ExperimentState import ExperimentState
from screens.welcome import input_screen, welcome_screen
from calibration.calibration_flow import calibration_screen
from practice_n_back import practice_n_back
from screens.instructions import instruction_screen
from maze_game_n import setup_level, is_maze_completed, complete_maze
from nback_task import load_sounds, play_sound
from data_logging import log_event, save_data_and_participant_info


def main():
    # 1. Initialize Pygame, LSL, etc.

    # Initialize Pygame and the mixer
    pygame.init()
    pygame.mixer.init()

    # Initialize the experiment state
    state = ExperimentState()

    # Initialize LSL stream
    state.outlet = pylsl.StreamOutlet(pylsl.StreamInfo("Trigger_Cog", "Markers", 1, 0, pylsl.cf_int32, "myuidw43536"))

    # Initialize the screen
    state.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    state.screen_width, state.screen_height = state.screen.get_size()  # Update dimensions to the full screen size
    pygame.display.set_caption("Maze experiment")



    # 2. Show welcome screen
    state.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    if not welcome_screen(state):
        return


    # 3. Collect participant details
    while True:
        user_details = input_screen(state)
        # check if user_details contain only numbers and its 3
        if user_details is not None and user_details['Serial Number'].isdigit() and len(
                user_details['Serial Number']) == 3:
            break

    # Log event for the start of the experiment
    log_event(6, datetime.now().timestamp(), state, cfg.stim_flag)


    # 4. Calibrations
    calibration_screen(state, cfg.stim_flag)


    # 5. Maze + N-back Setup
    object_sounds = load_sounds(
        os.path.join(os.path.dirname(__file__), "../../resources/back_sound/sound_object"),
        "object")

    animal_sounds = load_sounds(
        os.path.join(os.path.dirname(__file__), "../../resources/back_sound/sound_animal"),
        "animal")

    # 6. Practice N-Back
    practice_n_back(state, cfg, object_sounds, animal_sounds)

    # set up first level
    state.maze, state.cell_size, state.maze_background, state.offset_x, state.offset_y = (
        setup_level(state, cfg))


    # Instruction screen
    if not instruction_screen(state, cfg.stim_flag, user_details):
        exit()  # Exit if the user closes the window or presses ESCAPE

    state.screen = pygame.display.set_mode((state.screen_width, state.screen_height))

    # Create the maze background for the first level
    state.maze_background = pygame.Surface((state.screen_width, state.screen_height))

    # Draw the maze on the background surface
    state.maze_background.fill(cfg.BLACK)
    for y in range(state.maze.shape[0]):
        for x in range(state.maze.shape[1]):
            if state.maze[y, x] == 1:
                pygame.draw.rect(state.maze_background, cfg.WHITE, (x * state.cell_size, y * state.cell_size, state.cell_size, state.cell_size))

    # track the correctness of the response
    expected_response = False

    state.experiment_start_time = datetime.now() # Start time of the experiment
    state.stage_start_time = datetime.now() # Record the start time of the first stage

    # 7. Run the main game loop (or keep it inline)

    # Main game loop
    while state.running:
        # Reset the response flag for the new interval
        response_made = False  # Reset response flag for the new interval

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN:
                state.key_pressed = event.key
                if event.key == pygame.K_e:  # Handle 'E' key press to skip the stage
                    # Check if at least 5 minutes have passed since the stage started
                    elapsed_time = (datetime.now() - state.stage_start_time).total_seconds()
                    if elapsed_time >= 300:  # 300 seconds = 5 minutes
                        # Send unique trigger to signify that the user has stopped the maze in the middle.
                        log_event(19, datetime.now().timestamp(), state, cfg.stim_flag)  # Trigger ID 19
                        # Simulate the maze completion
                        complete_maze(state, cfg, user_details)
                        pygame.display.flip()
                        pygame.time.Clock().tick(30)  # Limit to 30 frames per second
                        continue  # Go to next iteration of the loop
                    else:
                        # Optionally provide feedback to the participant
                        print("You cannot skip the maze before 5 minutes have passed.")
            elif event.type == pygame.KEYUP:
                if event.key == state.key_pressed:
                    state.key_pressed = None

            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                response_made = True
                # Determine correctness and log event
                if expected_response:  # If a response was expected
                    log_event(2, datetime.now().timestamp(), state, cfg.stim_flag)  # Correct response, Trigger ID 2
                    print("correct response")
                else:
                    log_event(1, datetime.now().timestamp(), state, cfg.stim_flag)  # Incorrect response, Trigger ID 1
                    print("incorrect response")
                expected_response = False  # Reset for the next sound

        # Blit the maze background
        state.screen.blit(state.maze_background, (state.offset_x, state.offset_y))

        current_time = pygame.time.get_ticks()
        if state.key_pressed and current_time - state.last_move_time > cfg.cooldown_duration:
            # Move player based on key state
            if (state.key_pressed == pygame.K_LEFT and state.player_x - cfg.speed >= 0 and
                    state.maze[state.player_y, state.player_x - cfg.speed] == 0):
                state.player_x -= cfg.speed
            elif (state.key_pressed == pygame.K_RIGHT and state.player_x + cfg.speed < state.maze.shape[1] and
                  state.maze[state.player_y, state.player_x + cfg.speed] == 0):
                state.player_x += cfg.speed
            elif (state.key_pressed == pygame.K_UP and state.player_y - cfg.speed >= 0 and
                  state.maze[state.player_y - cfg.speed, state.player_x] == 0):
                state.player_y -= cfg.speed
            elif (state.key_pressed == pygame.K_DOWN and state.player_y + cfg.speed < state.maze.shape[0] and
                  state.maze[state.player_y + cfg.speed, state.player_x] == 0):
                state.player_y += cfg.speed
            #
            state.last_move_time = current_time

        # Sound playback logic
        if state.n_back_level > 0:
            current_time = pygame.time.get_ticks()
            if current_time >= state.sound_end_time:

                # Check if the player's missed the response
                if expected_response and not response_made:
                    log_event(3, datetime.now().timestamp(), state, cfg.stim_flag)  # Log missed response
                    print("missed response")
                    expected_response = False  # Reset for the next interval

                # Play the next sound
                if random.randint(1, 100) <= 40 and len(state.sound_sequence) >= state.n_back_level + 1:
                    chosen_sound_info = state.sound_sequence[-state.n_back_level]
                else:
                    chosen_list = random.choice([object_sounds, animal_sounds])
                    chosen_sound_info = random.choice(chosen_list)
                sound_duration = play_sound(chosen_sound_info, state)

                # Set the time when the next sound should be played
                state.sound_end_time = current_time + sound_duration + cfg.sound_delay

                # Determine if this sound requires a response based on N-BACK rule
                if (len(state.sound_sequence) >= state.n_back_level + 1 and
                        state.sound_sequence[-state.n_back_level - 1]['filename'] == chosen_sound_info['filename']):
                    if not state.animal_sound:
                        expected_response = True  # A response is expected for this sound
                    elif chosen_sound_info['type'] == 'animal':
                        expected_response = True

                response_made = False  # Reset response flag for the new interval

        # Draw the player
        pygame.draw.rect(state.screen, cfg.RED, ((state.player_x * state.cell_size + state.offset_x),
                                       (state.player_y * state.cell_size + state.offset_y), state.cell_size, state.cell_size))

        # Check if the maze is completed
        if is_maze_completed(state.player_x, state.player_y, state.maze):
            complete_maze(state, cfg, user_details)

        pygame.display.flip()
        pygame.time.Clock().tick(30)  # Limit to 30 frames per second

    # 8. Save data
    save_data_and_participant_info(state, cfg.stim_flag, user_details)
    pygame.quit()

if __name__ == "__main__":
    main()