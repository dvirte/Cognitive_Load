import wave
import os
import pygame

def get_sound_duration(file_path):
    with wave.open(file_path, 'rb') as wav_file:  # Open the WAV file in read-binary mode
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
    return int(duration * 1000)  # Convert to milliseconds

def load_sounds(folder, sound_type):
    sounds = []
    for filename in os.listdir(folder):
        if filename.endswith('.wav'):  # Adjust the format as needed
            path = os.path.join(folder, filename)
            sound = pygame.mixer.Sound(path)
            duration = get_sound_duration(path)
            sound_info = {'sound': sound, 'type': sound_type,
                          'filename': filename, 'duration': duration}
            sounds.append(sound_info)
    return sounds

def play_sound(sound_info, state):
    pygame.mixer.Sound.play(sound_info['sound'])
    state.sound_sequence.append(sound_info)  # Use a unique identifier for the sound
    return sound_info['duration']