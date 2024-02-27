import matplotlib.pyplot as plt
import numpy as np
import random
from queue import Queue


def create_maze_old(dim, branch_prob=0.3):
    # Initialize the maze as a grid filled with “1”s, representing walls. Represent cells as “0”s.
    # Create a grid filled with walls
    maze = np.ones((dim * 2 + 1, dim * 2 + 1))

    # Define the starting point
    x, y = (0, 0)
    maze[2 * x + 1, 2 * y + 1] = 0

    # Initialize the stack with the starting point
    stack = [(x, y)]
    while len(stack) > 0:
        x, y = stack[-1]

        # Define possible directions
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)

        moved = False
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if nx >= 0 and ny >= 0 and nx < dim and ny < dim and maze[2 * nx + 1, 2 * ny + 1] == 1:
                maze[2 * nx + 1, 2 * ny + 1] = 0
                maze[2 * x + 1 + dx, 2 * y + 1 + dy] = 0
                stack.append((nx, ny))
                moved = True
                break
        if not moved or random.random() < branch_prob:
            stack.pop()

    # Create an entrance and an exit
    maze[1, 0] = 0
    maze[-2, -1] = 0

    return maze


def is_path_old(maze, start, end):
    """Check if there is a path from start to end using BFS."""
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    visited = set()
    queue = Queue()
    queue.put(start)
    while not queue.empty():
        x, y = queue.get()
        if (x, y) == end:
            return True
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] == 0 and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.put((nx, ny))
    return False


def create_valid_maze_old(dim, branch_prob=0.3):
    """Create a maze and validate it has a path from entrance to exit."""
    while True:
        maze = create_maze(dim, branch_prob)
        entrance = (1, 0)
        exit = (maze.shape[0] - 2, maze.shape[1] - 1)
        if is_path(maze, entrance, exit):
            return maze


def direct_path_check(maze, start, end, entrance, exit):
    """Check if adding a loop between start and end points simplifies the maze too much."""
    original_path_length = bfs_shortest_path(maze, entrance, exit)

    # Temporarily add the loop
    maze[start[0], start[1]] = 0
    maze[end[0], end[1]] = 0

    new_path_length = bfs_shortest_path(maze, entrance, exit)

    # Remove the loop
    maze[start[0], start[1]] = 1
    maze[end[0], end[1]] = 1

    # Define what "significantly shorter" means; e.g., reduce path length by more than 10%
    if new_path_length <= original_path_length * 0.9:
        return True
    else:
        return False


def create_valid_maze(dim, branch_prob=0.3):
    """Create a maze and validate it has a path from entrance to exit."""
    while True:
        maze = create_maze(dim, branch_prob)
        entrance = (1, 0)  # Define entrance here, as before
        exit = (maze.shape[0] - 2, maze.shape[1] - 1)  # Define exit based on maze dimensions
        draw_maze(maze)
        print(is_path_old(maze, entrance, exit))
        print(is_path(maze, entrance, exit))
        if is_path(maze, entrance, exit):
            add_loops(maze, entrance, exit, attempts=dim * 2)  # Add complexity after validating the path
            return maze


def create_maze(dim, branch_prob=0.3, min_branch_length=3, max_branch_length=7):
    maze = np.ones((dim * 2 + 1, dim * 2 + 1))
    x, y = 1, 1  # Start from (1, 1) to ensure the starting point is inside the maze bounds
    maze[x, y] = 0
    stack = [(x, y)]

    while stack:
        x, y = stack[-1]
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)

        moved = False
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < dim * 2 and 0 <= ny < dim * 2 and maze[nx + dx, ny + dy] == 1:
                # Ensure the next step in the same direction is within bounds and not already part of the path
                maze[nx, ny] = 0
                maze[nx + dx, ny + dy] = 0
                stack.append((nx + dx, ny + dy))
                moved = True
                break  # Break after moving to prevent checking other directions unnecessarily

        if not moved:
            stack.pop()

    return maze


def add_loops(maze, entrance, exit, attempts=10):
    """
    Add loops to the maze to increase complexity without creating large open areas.

    :param maze: The numpy array representing the maze.
    :param entrance: Tuple of entrance coordinates (x, y).
    :param exit: Tuple of exit coordinates (x, y).
    :param attempts: Number of attempts to add loops to the maze.
    """
    for _ in range(attempts):
        # Choose a random wall in the maze
        x, y = random.randint(1, maze.shape[0] - 2), random.randint(1, maze.shape[1] - 2)

        if maze[x, y] == 1:  # If it's a wall, attempt to add a loop
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            random.shuffle(directions)

            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                # Ensure the new point is within the maze and adjacent to an open path
                if 0 < nx < maze.shape[0] - 1 and 0 < ny < maze.shape[1] - 1 and maze[nx, ny] == 0:
                    # Check if converting the wall to a path creates a loop without forming a large open area
                    if not forms_large_open_area(maze, x, y):
                        maze[x, y] = 0  # Convert wall to path, forming a loop
                        break  # Exit after successfully adding a loop


def forms_large_open_area(maze, x, y):
    """
    Check if removing a wall at (x, y) will create a large open area in the maze.

    :param maze: The numpy array representing the maze.
    :param x: X-coordinate of the wall.
    :param y: Y-coordinate of the wall.
    :return: True if removing the wall creates a large open area, False otherwise.
    """
    # Check adjacent cells; if there are too many open paths, it might form a large open area
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    open_paths = sum(maze[x + dx, y + dy] == 0 for dx, dy in directions if
                     0 <= x + dx < maze.shape[0] and 0 <= y + dy < maze.shape[1])

    # Arbitrarily define "too many open paths" as more than 3 adjacent open paths; adjust as needed
    return open_paths > 3


def bfs_shortest_path(maze, start, end):
    """Find the shortest path in the maze from start to end using BFS."""
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    visited = set()
    queue = Queue()
    queue.put((start, 0))  # Each queue entry is a tuple (position, distance)

    while not queue.empty():
        (x, y), distance = queue.get()
        if (x, y) == end:
            return distance

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] == 0 and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.put(((nx, ny), distance + 1))
    return float('inf')  # Return infinity if no path is found


def is_path(maze, start, end):
    # return True
    """Check if there is a path from start to end using BFS."""
    # Check if start or end points are walls
    if maze[start[0], start[1]] == 1 or maze[end[0], end[1]] == 1:
        return False  # No path if start or end is a wall

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    visited = set()
    queue = Queue()
    queue.put(start)
    while not queue.empty():
        x, y = queue.get()
        if (x, y) == end:
            return True
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] == 0 and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.put((nx, ny))
    return False


def draw_maze(maze, path=None):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Set the border color to white
    fig.patch.set_edgecolor('white')
    fig.patch.set_linewidth(0)

    ax.imshow(maze, cmap=plt.cm.binary, interpolation='nearest')

    # Draw the solution path if it exists
    if path is not None:
        x_coords = [x[1] for x in path]
        y_coords = [y[0] for y in path]
        ax.plot(x_coords, y_coords, color='red', linewidth=2)

    ax.set_xticks([])
    ax.set_yticks([])

    # Draw entry and exit arrows
    ax.arrow(0, 1, .4, 0, fc='green', ec='green', head_width=0.3, head_length=0.3)
    ax.arrow(maze.shape[1] - 1, maze.shape[0] - 2, 0.4, 0, fc='blue', ec='blue', head_width=0.3, head_length=0.3)

    plt.show()


if __name__ == "__main__":
    # Define the maze parameters
    dim = 20  # Dimension of the maze
    branch_prob = 0.3  # Probability of branching

    # Generate the maze
    maze = create_valid_maze(dim, branch_prob=branch_prob)

    draw_maze(maze)
