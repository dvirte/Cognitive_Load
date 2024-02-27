import numpy as np
from pyamaze import maze
import matplotlib.pyplot as plt
from collections import deque


def bfs(maze, start, end):
    rows, cols = maze.shape
    visited = np.full((rows, cols), False)
    queue = deque([(start, 0)])  # (position, distance)

    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        (x, y), dist = queue.popleft()

        # Exit found
        if (x, y) == end:
            return dist

        # Mark as visited
        visited[x, y] = True

        # Explore neighbors
        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # Check if within bounds, not a wall, and not visited
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx, ny] == 0 and not visited[nx, ny]:
                queue.append(((nx, ny), dist + 1))

    # Return -1 if no path is found
    return -1


def dfs(maze, start, end, visited=None, path_length=0, shortest_path=[float('inf')]):
    if visited is None:
        visited = set()

    x, y = start
    if start == end:  # Base case: end reached
        shortest_path[0] = min(shortest_path[0], path_length)
        return

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    visited.add(start)  # Mark the current cell as visited

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if maze.shape[0] > nx >= 0 == maze[nx, ny] and 0 <= ny < maze.shape[1] and (nx, ny) not in visited:
            dfs(maze, (nx, ny), end, visited, path_length+1, shortest_path)

    visited.remove(start)  # Remove current cell from visited set to allow other paths

    if start == (1, 0):  # If we're returning from the initial call, return the shortest path length found
        return shortest_path[0] if shortest_path[0] != float('inf') else -1


def add_holes_and_increase_complexity(maze_array, max_holes=4, prob=1):
    rows, cols = maze_array.shape
    entrance = (1, 0)
    exit_path = (rows - 2, cols-1)
    original_length = dfs(maze_array, entrance, exit_path)

    holes_added = 0
    wall_positions = [(r, c) for r in range(2, rows - 2, 2) for c in range(2, cols - 2, 2) if maze_array[r, c] == 1]
    while holes_added < max_holes:

        # Randomly select a wall (1) that is not on the perimeter
        if not wall_positions:
            break
        row, col = wall_positions[np.random.randint(len(wall_positions))]
        stop = False

        # Check id square is formed by walls
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:  # Diagonal directions
            if row+dr < 0 or row+dr >= rows or col+dc < 0 or col+dc >= cols:
                continue
            if maze_array[row+dr, col] == 0 and maze_array[row, col+dc] == 0 and maze_array[row+dr, col+dc] == 0:
                # remove wall from list of candidates and go back th while loop
                wall_positions.remove((row, col))
                stop = True
                break

        # Skip to next iteration if a square is formed
        if stop:
            continue

        # Check if the wall have more than 1 neighbor:
        neighbors = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if row+dr < 0 or row+dr >= rows or col+dc < 0 or col+dc >= cols:
                continue
            if maze_array[row+dr, col+dc] == 0:
                neighbors += 1
        if neighbors < 2:
            wall_positions.remove((row, col))
            continue

        # Drill a hole
        maze_array[row, col] = 0
        new_length = dfs(maze_array, entrance, exit_path)

        if new_length <= original_length*prob:
            maze_array[row, col] = 1  # Revert if no increase in complexity
            wall_positions.remove((row, col))
        else:
            holes_added += 1
            original_length = new_length  # Update path length for next iteration

    return maze_array


def remove_path_and_increase_complexity(maze_array):
    rows, cols = maze_array.shape
    entrance = (1, 0)
    exit_path = (rows - 2, cols - 1)
    original_length = bfs(maze_array, entrance, exit_path)

    while not path_cells:
        # Randomly select a cell from the path
        path_cells = [(r, c) for r in range(1, rows - 1) for c in range(1, cols - 1) if maze_array[r, c] == 0]
        row, col = path_cells[np.random.randint(len(path_cells))]

        # Remove cell from path
        maze_array[row, col] = 1
        new_length = bfs(maze_array, entrance, exit_path)

        if new_length >= original_length:
            original_length = new_length  # Update path length for next iteration
        else:
            maze_array[row, col] = 0  # Revert if no increase in complexity

        # Remove cell from list of candidates
        path_cells.remove((row, col))

    return maze_array


def maze_to_ndarray(dim, max_holes, prob):

    # Create a maze using pyamaze
    m = maze(dim, dim)
    m.CreateMaze()

    # Initialize ndarray with zeros, each cell is surrounded by walls, hence +1 for outer walls
    arr_size = (m.rows * 2 + 1, m.cols * 2 + 1)
    arr = np.ones(arr_size, dtype=np.uint8)

    # Mark passages in arr by interpreting m.maze_map
    for cell, directions in m.maze_map.items():
        row, col = cell
        # Convert cell coordinate to array indices, accounting for walls
        arr_row, arr_col = row * 2 - 1, col * 2 - 1
        arr[arr_row, arr_col] = 0  # Mark the cell itself as open
        if 'E' in directions and directions['E']:
            arr[arr_row, arr_col + 1] = 0  # East passage
        if 'W' in directions and directions['W']:
            arr[arr_row, arr_col - 1] = 0  # West passage
        if 'S' in directions and directions['S']:
            arr[arr_row + 1, arr_col] = 0  # South passage
        if 'N' in directions and directions['N']:
            arr[arr_row - 1, arr_col] = 0  # North passage

    # Create an entrance and an exit
    arr[1, 0] = 0
    arr[-2, -1] = 0

    add_holes_and_increase_complexity(arr, max_holes, prob)

    return arr


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