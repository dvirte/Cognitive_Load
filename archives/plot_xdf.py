import pyxdf as pyxdf
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings
mpl.use('Tkagg')
def load_data(xdf_file):
    """
    Load XDF file and prepare data dictionary
    """
    print(f"Loading XDF file: {xdf_file}")
    streams, fileheader = pyxdf.load_xdf(xdf_file)

    # Process streams into a dictionary
    storage_dict = {}
    for stream in streams:
        stream_name = stream['info']['name'][0]
        current_time_stamps = np.array(stream['time_stamps'])
        current_time_series = np.array(stream['time_series'])

        storage_dict[stream_name] = {
            'time_stamps': current_time_stamps,
            'time_series': current_time_series
        }

    return storage_dict


class SignalViewer:
    def __init__(self, storage_dict, window_size=12):
        """
        Interactive signal viewer for electrode and audio data with trigger markers

        Args:
            storage_dict: Dictionary containing data streams
            window_size: Initial window size in seconds
        """
        self.storage_dict = storage_dict
        self.window_size = window_size

        # Extract data
        self.electrode_data = storage_dict['ElectrodeStream']
        self.audio_data = storage_dict['MyAudioStream']
        self.trigger_data = storage_dict['Sync_Check']

        # Extract channel 5 from electrode data (index 5 since 0-indexed)
        self.electrode_ch5 = self.electrode_data['time_series'][:, 5]

        # Get time limits
        self.electrode_times = self.electrode_data['time_stamps']
        self.audio_times = self.audio_data['time_stamps']
        self.trigger_times = self.trigger_data['time_stamps']
        self.trigger_values = self.trigger_data['time_series']

        # For simplicity, assuming all streams start around the same time
        self.min_time = min(np.min(self.electrode_times), np.min(self.audio_times))

        # Normalize times to start at 0
        self.electrode_times = self.electrode_times - self.min_time
        self.audio_times = self.audio_times - self.min_time
        self.trigger_times = self.trigger_times - self.min_time

        # Get max time for the recording
        self.max_time = max(np.max(self.electrode_times), np.max(self.audio_times))

        # Initial view window
        self.current_pos = 0

        # Create the plot
        self.create_plot()

    def create_plot(self):
        """Create the interactive matplotlib plot"""
        # Create figure and subplots
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        plt.subplots_adjust(bottom=0.25)  # Make room for slider

        # Initial plot range
        self.update_plot_range()

        # Create slider for navigating
        self.ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03], facecolor='lightgoldenrodyellow')
        self.time_slider = Slider(
            self.ax_slider, 'Time (s)',
            0,
            max(0, self.max_time - self.window_size),
            valinit=self.current_pos
        )
        self.time_slider.on_changed(self.update_slider)

        # Add buttons for navigation
        self.ax_prev = plt.axes([0.15, 0.05, 0.15, 0.03])
        self.ax_next = plt.axes([0.7, 0.05, 0.15, 0.03])
        self.btn_prev = Button(self.ax_prev, 'Previous Trigger')
        self.btn_next = Button(self.ax_next, 'Next Trigger')
        self.btn_prev.on_clicked(lambda event: self.jump_to_trigger('prev'))
        self.btn_next.on_clicked(lambda event: self.jump_to_trigger('next'))

        # Add window size buttons
        self.ax_win5 = plt.axes([0.35, 0.05, 0.07, 0.03])
        self.ax_win12 = plt.axes([0.44, 0.05, 0.07, 0.03])
        self.ax_win30 = plt.axes([0.53, 0.05, 0.07, 0.03])
        self.ax_win60 = plt.axes([0.62, 0.05, 0.07, 0.03])

        self.btn_win5 = Button(self.ax_win5, '5s')
        self.btn_win12 = Button(self.ax_win12, '12s')
        self.btn_win30 = Button(self.ax_win30, '30s')
        self.btn_win60 = Button(self.ax_win60, '60s')

        self.btn_win5.on_clicked(lambda event: self.set_window_size(5))
        self.btn_win12.on_clicked(lambda event: self.set_window_size(12))
        self.btn_win30.on_clicked(lambda event: self.set_window_size(30))
        self.btn_win60.on_clicked(lambda event: self.set_window_size(60))

        # Set titles and labels
        self.ax1.set_title('ElectrodeStream (Channel 5)')
        self.ax2.set_title('MyAudioStream')
        self.ax2.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Amplitude')
        self.ax2.set_ylabel('Amplitude')

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)

    def update_plot_range(self):
        """Update the plot based on the current position and window size"""
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()

        # Set xlim for both plots
        x_min = self.current_pos
        x_max = self.current_pos + self.window_size
        self.ax1.set_xlim(x_min, x_max)
        self.ax2.set_xlim(x_min, x_max)

        # Filter data for the visible window
        electrode_mask = (self.electrode_times >= x_min) & (self.electrode_times <= x_max)
        audio_mask = (self.audio_times >= x_min) & (self.audio_times <= x_max)

        # Plot electrode data (channel 5)
        self.ax1.plot(
            self.electrode_times[electrode_mask],
            self.electrode_ch5[electrode_mask],
            'b-', linewidth=1
        )

        # Plot audio data
        self.ax2.plot(
            self.audio_times[audio_mask],
            self.audio_data['time_series'][audio_mask],
            'g-', linewidth=1
        )

        # Plot trigger lines
        visible_triggers = (self.trigger_times >= x_min) & (self.trigger_times <= x_max)
        for t in self.trigger_times[visible_triggers]:
            self.ax1.axvline(x=t, color='r', linestyle='-', alpha=0.7)
            self.ax2.axvline(x=t, color='r', linestyle='-', alpha=0.7)

        # Set titles to show current window
        self.ax1.set_title(f'ElectrodeStream (Channel 5) - Viewing {x_min:.1f}s to {x_max:.1f}s')
        self.ax2.set_title(f'MyAudioStream - {np.sum(visible_triggers)} triggers visible')

        # Update the figure
        self.fig.canvas.draw_idle()

    def update_slider(self, val):
        """Callback for slider movement"""
        self.current_pos = val
        self.update_plot_range()

    def jump_to_trigger(self, direction):
        """Jump to the next or previous trigger"""
        middle_time = self.current_pos + (self.window_size / 2)

        if direction == 'next':
            # Find the next trigger after the current middle point
            next_triggers = self.trigger_times[self.trigger_times > middle_time]
            if len(next_triggers) > 0:
                next_trigger = next_triggers[0]
                self.current_pos = max(0, next_trigger - (self.window_size / 2))
                self.time_slider.set_val(self.current_pos)
        else:  # previous
            # Find the previous trigger before the current middle point
            prev_triggers = self.trigger_times[self.trigger_times < middle_time]
            if len(prev_triggers) > 0:
                prev_trigger = prev_triggers[-1]
                self.current_pos = max(0, prev_trigger - (self.window_size / 2))
                self.time_slider.set_val(self.current_pos)

    def set_window_size(self, size):
        """Change the window size"""
        middle = self.current_pos + (self.window_size / 2)
        self.window_size = size
        self.current_pos = max(0, middle - (self.window_size / 2))

        # Update slider max value
        self.time_slider.valmax = max(0, self.max_time - self.window_size)
        self.time_slider.ax.set_xlim(self.time_slider.valmin, self.time_slider.valmax)

        # Make sure current position is valid
        if self.current_pos > self.time_slider.valmax:
            self.current_pos = self.time_slider.valmax

        self.time_slider.set_val(self.current_pos)
        self.update_plot_range()

    def show(self):
        """Display the interactive plot"""
        plt.show()


# Example usage:
if __name__ == "__main__":
    # Path to the XDF file
    current_path = os.getcwd()
    parent_path = os.path.join(current_path, '..')
    xdf_name = os.path.join(parent_path, 'data', 'test_data', 'stim_trigger_mic', '01.xdf')

    # Load the data
    try:
        storage_dict = load_data(xdf_name)
        # Create and show the interactive viewer
        viewer = SignalViewer(storage_dict)
        viewer.show()
    except FileNotFoundError:
        print(f"File not found: {xdf_name}")
        print("You may need to adjust the file path.")
