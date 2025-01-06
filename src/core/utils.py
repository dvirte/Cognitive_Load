import pygame

def display_text(state, text, font_size=50, color=(255, 255, 255), y_offset=0):

    screen_width = state.screen_width
    screen_height = state.screen_height

    font = pygame.font.Font(None, font_size)
    text_surface = font.render(text, True, color)
    # Set the text position dynamically with a vertical offset
    rect = text_surface.get_rect(center=(screen_width // 2, (screen_height // 2) + y_offset))
    state.screen.blit(text_surface, rect)

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