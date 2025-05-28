# Trigger ID Documentation (For Documentation Purposes Only)
# trigger_id_docs = {
#     1: 'Indicates that the participant incorrectly pressed the spacebar
#           when the stimulus did not match the N-BACK condition.',
#     2: 'Indicates that the participant pressed the spacebar correctly
#           in response to a stimulus matching the N-BACK condition.',
#     3: 'Logs instances where a response was expected (the stimulus matched the N-BACK condition)
#           but the participant did not press the spacebar, indicating a missed response.',
#     4: 'Indicates the start of a new level in the experiment.',
#     5: 'Indicates the end of the experiment.'
#     6: 'Indicates the start of the experiment.'
#     7: 'Indicates the start of the N-BACK Instructions.'
#    10: 'Indicates when the subject has high error rates'
#    11: 'Indicates start of the NASA-TLX rating'
#    13: 'Indicates for calibration: start blinking'
#    14: 'Indicates for calibration: start jaw stiffness'
#    15: 'Indicates for calibration: start frowning'
#    16: 'Indicates for calibration: start yawning'
#    17: 'Indicates for calibration: start calibration'
#    18: 'Indicates for calibration: end calibration'
#    19: 'Indicates that the participant skipped the maze in the middle.'
#    20: 'Indicates that a new maze is started because the previous maze was completed in less than 1.5 minutes.'
#    21: 'Calibration step: start of white dot with eyes open',
#    22: 'Calibration step: beep to close eyes',
#    23: 'Calibration step: end of white dot calibration',
#    24: 'Jaw Calibration: Clenching',
#    25: 'Jaw Calibration: Jaw opening',
#    26: 'Jaw Calibration: Chewing',
#    27: 'Jaw Calibration: Resting',
#    28: 'Synchronization Calibration: Start',
#    29: 'Synchronization Calibration: Single sound',
#    30: 'Synchronization Calibration: End',
#    31: 'n-back training phase 1',
#    32: 'n-back training phase 2',
#    33: 'n-back training phase 3',
# }


maze_size = 5  # Initial maze size
current_threshold = 0.3  # Initial error threshold
maze_complexity = 0.8  # Initial maze complexity
amount_of_levels = 10  # Amount of levels to play before starting the N-back task
time_of_experiment = 10  # Time in minutes for the experiment

# Initialize sound delay
sound_delay = 500  # Initial delay between sounds in milliseconds (0.5 seconds)
min_sound_delay = 500  # Minimum delay (0.5 seconds)
max_sound_delay = 5000  # Maximum delay (5 seconds)

# Maze parameters
dim = 30  # Size of the maze

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Player speed (number of cells per frame)
speed = 1

cooldown_duration = 50  # milliseconds

stim_flag = False