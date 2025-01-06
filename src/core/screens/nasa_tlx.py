import pygame
from datetime import datetime
import random
# from src.core.data_logging import log_event

def nasa_tlx_rating_screen(state, stim_flag):
    from src.core.data_logging import log_event
    # Log event for the start of the NASA-TLX rating
    log_event(11, datetime.now().timestamp(), state, stim_flag)  # Trigger ID 11

    # Define screen properties
    screen = pygame.display.set_mode((state.screen_width, state.screen_height))
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
        screen.blit(title_surface, (state.screen_width // 2 - title_surface.get_width() // 2, 50))  # 50px padding from top

        # Calculate the vertical centering for the scales, increase space between scales
        total_content_height = len(categories) * 120  # Increase spacing between categories
        start_y_position = (state.screen_height - total_content_height) // 2 + 50  # Add margin for title

        # Draw scales and labels
        for i, category in enumerate(categories):
            y_position = start_y_position + i * 120  # Adjusted spacing between questions
            text_label = small_font.render(category, True, WHITE)

            # Center the text label relative to the screen width
            text_x_position = state.screen_width // 2 - text_label.get_width() // 2
            screen.blit(text_label, (text_x_position, y_position))

            # Draw the scale line centered
            line_start_x = state.screen_width // 2 - 350  # Increase line length for the scale
            line_end_x = state.screen_width // 2 + 350
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
        start_y_position = (state.screen_height - total_content_height) // 2 + 50

        for i, category in enumerate(categories):
            y_position = start_y_position + i * 120
            line_start_x = state.screen_width // 2 - 350
            line_end_x = state.screen_width // 2 + 350

            # The clickable area for each scale
            clickable_area_start_y = y_position + 30  # Adjust based on new scale position
            clickable_area_end_y = y_position + 50

            if line_start_x <= pos[0] <= line_end_x and clickable_area_start_y <= pos[1] <= clickable_area_end_y:
                scale_values[category] = round((pos[0] - line_start_x) / (700 / 19))  # Adjust for new scale width
                return

    # Position and draw the "Continue" button
    continue_button = pygame.Rect(state.screen_width // 2 - 50, state.screen_height - 120, 120, 50)  # Move button up a bit

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
        state.stage_performance[-1][scale] = scale_values[scale]


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

