class ExperimentState:
    def __init__(self):
        self.experiment_data = []  # Initialize an empty list to store event data
        self.level_list = [0, 0, 0]  # List to keep track of which aspect to increase next
        self.performance_ratios = {'TP': [], 'FP': [], 'start': [], 'end': []}  # Initialize performance ratios
        self.stage_performance = []  # Initialize list to store high error performance
        self.path_of_maze = []  # Initialize list to store the path of the maze
        self.baseline_maze = 0  # Play 10 levels of maze without any n-back task
        self.animal_sound = True  # The n-back task only takes into account if the sound played is from animals and not inanimate
        self.middle_calibration = True  # Flag to indicate if the calibration is in the middle of the experiment
        self.sound_sequence = []  # Reset for the new level
        self.sound_end_time = 0  # Reset the sound end time
        self.last_move_time = 0  # track the last move time
        self.n_back_level = 0  # Initialize the n-back level to 0
        self.key_pressed = None  # Initialize the key pressed to None
        self.stage_start_time = None # Initialize the start time of the stage

        # Player position - starting at the entrance of the maze
        self.player_x, self.player_y = 0, 1  # Adjusted to start at the maze entrance

        # Screen dimensions
        self.screen_width = None
        self.screen_height = None
        self.screen = None

        # Outlet for LSL stream
        self.outlet = None

        # Flag to indicate if the experiment is running
        self.running = True

        # Maze and cell size
        self.maze = None
        self.cell_size = None
        self.maze_background = None
        self.offset_x = None
        self.offset_y = None