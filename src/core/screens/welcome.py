import pygame
from src.core.utils import draw_text, display_text

def welcome_screen(state):

    screen = state.screen
    screen_width = state.screen_width
    screen_height = state.screen_height

    running = True
    while running:
        screen.fill((0, 0, 0))
        display_text(state, "Welcome to the Experiment", font_size=50, y_offset=-100)  # Move this line up
        display_text(state, "Press Enter to continue", font_size=40, y_offset=20)       # Centered
        display_text(state, "or Escape to exit", font_size=40, y_offset=90)           # Move this line down

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
                pygame.quit()
                return False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                running = False

        pygame.display.flip()

    return True

def input_screen(state):
    screen = state.screen
    screen_width = state.screen_width
    screen_height = state.screen_height

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