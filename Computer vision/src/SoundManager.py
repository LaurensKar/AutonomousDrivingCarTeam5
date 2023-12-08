import subprocess
import time


class SoundManager:
    def __init__(self):
        self.sounds = {
            "lane_departure": ['./sounds/lane_departure.mp3', 1000, 1],
            "hands_on_wheel": ['./sounds/hands_on_steering_wheel.mp3', 200, 2],
            "autopilot_engaged": ['./sounds/autosteer_enabled.wav', 100, 3],
            "autopilot_disabled": ['./sounds/autosteer_disabled.mp3', 100, 3],
            "epb_disable": ['./sounds/epb_disable.mp3', 100, 3],
            "crash": ['./sounds/crash.mp3', 1000, 3],
            "collision": ['./sounds/forward_collision_warning.mp3', 1000, 3]
        }

        # Initialize with a list of [0, 0] for each key
        self.last_played = {key: [0, 0] for key in self.sounds.keys()}

    def play_lane_departure_sound(self):
        self.play_sound("lane_departure")

    def play_hands_on_wheel_sound(self):
        self.play_sound("hands_on_wheel")

    def play_autopilot_enabled(self):
        self.play_sound("autopilot_engaged")

    def play_parking_brake_disabled(self):
        self.play_sound("epb_disable")

    def play_autopilot_disabled(self):
        self.play_sound("autopilot_disabled")

    def play_crash(self):
        self.play_sound("crash")

    def play_collision_warning(self):
        self.play_sound("collision")

    def play_sound(self, sound_key):
        sound_path, delay, priority = self.sounds.get(sound_key, [None, None, None])
        now = time.time()
        last_played, last_priority = self.last_played.get(sound_key, [0, 0])

        if sound_path and (now - last_played) > delay / 1000.0 and priority >= last_priority:
            subprocess.Popen(['afplay', sound_path])
            # Save the current time and priority as the last played details for this sound key
            self.last_played[sound_key] = [now, priority]
        else:
            print(f"Cannot play sound for key: {sound_key}. Delay not met.")
