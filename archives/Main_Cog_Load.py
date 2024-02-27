# Import necessary libraries
from psychopy import visual, event, core
import asyncio
import numpy as np
# import sounddevice as sd
import os
import glob
import random
# import pylsl
import os
# import pylsl

# Create a window in fullscreen mode
win = visual.Window(fullscr=True, color='white', units='norm', allowGUI=False)

# Hide the cursor
win.mouseVisible = False

# Create first text
text = visual.TextStim(win, text='Welcome to the experiment \n\nPress any key to continue', color='black', height=0.1)

# Draw the text to the window and flip the window to show the text
text.draw()
win.flip()

# Wait for a key press
event.waitKeys()




