from collections import deque

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from src.Logger import setup_logger
from src.SoundManager import SoundManager


class Autosteer:
    # Constant attributes
    LOST_LINE_OFFSET = 0.005
    LOST_LINE_SPEED = 0.01
    NO_LINES_SPEED = -10000
    STEERING_BOOST = 3.7

    def __init__(self, n=10, alpha=0.69, history_length=5):

        self.soundmanager = SoundManager()
        self.logger = setup_logger()

        self.current_steering_angle = 0
        self.n = n
        self.alpha = alpha
        self.max_angle = 100
        self.min_angle = -100
        self.throttle = 10
        self.offset = 0
        self.slope_threshold = 1  # Adjust as needed
        self.attempts = 0

        self.left_line = []
        self.right_line = []
        self.detected_left_line = []
        self.detected_right_line = []

        self.history_left_lines = deque(maxlen=history_length)  # Stores the history of left lines detected
        self.history_right_lines = deque(maxlen=history_length)  # Stores the history of right lines detected
        self.fig, self.axs = None, None

    def detect_left_and_right_lines(self, lines):
        if len(lines) > 10:
            self.soundmanager.play_collision_warning()

        if lines is not None:
            slopes_lines = [((y2 - y1) / (x2 - x1) if (x2 != x1) else float('inf'), line)
                            for line in lines for x1, y1, x2, y2 in line]

            # Split into left and right line
            self.detected_left_line = [line for slope, line in slopes_lines if slope < 0]
            self.detected_right_line = [line for slope, line in slopes_lines if slope > 0]

    def save_lines_into_memory_and_update_throttle(self):
        """
        This function will update the throttle based on the detected lines.
        """
        self.offset = 0  # Reset steering offset

        # Any line detected?
        if self.detected_left_line or self.detected_right_line:

            # --- Both lines detected ---
            if self.detected_left_line and self.detected_right_line:
                self.throttle = 0.26  # Insane speed

            # --- Left line is detected ---
            if self.detected_left_line:
                self.left_line = self.detected_left_line
                self.history_left_lines.append(self.left_line)  # Save into memory
            elif len(self.history_left_lines) > 0:
                self.left_line = self.history_left_lines[-1]
                print("No left line was detected. Using the last known left line.")
                self.logger.warning("No left line was detected")
                self.offset = Autosteer.LOST_LINE_OFFSET  # Turn right a bit to find the right lane again
                self.throttle = Autosteer.LOST_LINE_SPEED  # Brake until you find both lanes again
                self.soundmanager.play_lane_departure_sound()

            # --- Right line is detected ---
            if self.detected_right_line:
                self.right_line = self.detected_right_line
                self.history_right_lines.append(self.detected_right_line)  # Save into memory
            elif len(self.history_right_lines) > 0:
                self.right_line = self.history_right_lines[-1]
                print("No right line was detected. Using the last known left line.")
                self.logger.warning("No right line was detected")
                self.offset = -Autosteer.LOST_LINE_OFFSET  # Turn left a bit to find the left line again
                self.throttle = Autosteer.LOST_LINE_SPEED  # Brake until you find both lanes again
                self.soundmanager.play_lane_departure_sound()

        # --- Nothing is detected, abort, and take over please ---      
        else:
            print("No lines detected, using history lines to come to a stop")
            self.soundmanager.play_hands_on_wheel_sound()
            self.logger.error("No lines detected, using history lines to come to a stop")
            self.attempts += 1
            print(f"Attempt {self.attempts}")
            self.logger.warning(f"Attempt {self.attempts}")
            self.left_line = self.history_left_lines[-1]
            self.right_line = self.history_right_lines[-1]
            self.throttle = Autosteer.NO_LINES_SPEED

    def steering_calculation(self):
        # Calculation
        if self.detected_left_line or self.detected_right_line:
            # Check if left line is detected
            if self.detected_left_line:
                # Calculate intersection point for left lines
                left_intersection = np.mean([np.array(line).mean(axis=0) for line in self.detected_left_line],
                                            axis=0) if self.detected_left_line else None
            else:
                left_intersection = None

            # Check if right line is detected
            if self.detected_right_line:
                # Calculate intersection point for right lines
                right_intersection = np.mean([np.array(line).mean(axis=0) for line in self.detected_right_line],
                                             axis=0) if self.detected_right_line else None
            else:
                right_intersection = None

            # Check if both left and right intersections are valid
            if left_intersection is not None and right_intersection is not None:
                # Check if mean_intersection has more than one element
                if len(left_intersection) > 1 and len(right_intersection) > 1:
                    # Calculate mean position between left and right intersections
                    mean_intersection = np.mean([left_intersection, right_intersection], axis=0)

                    # Calculate the target steering angle based on the slope of the line connecting the intersections
                    target_angle = (np.arctan((mean_intersection[3] - mean_intersection[1]) / (
                            mean_intersection[2] - mean_intersection[0])) * (180.0 / np.pi)) * Autosteer.STEERING_BOOST

                    # Smoothly update the current steering angle using a weighted average
                    self.current_steering_angle = self.alpha * self.current_steering_angle + (
                            1 - self.alpha) * target_angle
                    self.current_steering_angle = self.current_steering_angle / self.max_angle

                else:
                    print("No valid intersection points calculated, maintaining current angle.")
            else:
                print("No intersection points calculated")

        if self.current_steering_angle < 0.004:
            self.throttle = 0.30  # Speed boost
            # play_sounds('./sounds/plaid_reactions.mp4')

        if self.current_steering_angle > 0.015:
            self.throttle = 0.15
            print("Small turn, restricting speed")

        if self.current_steering_angle > 0.040:
            self.throttle = 0.05
            # play_sounds('./sounds/speed_assist_warning.mp3')

        if -0.01 < self.throttle < 0.01:
            print("Zero throttle")
            self.soundmanager.play_crash()

        if self.attempts > 2:
            print('Autopilot aborted')
            self.logger.error("Autopilot aborted, unavailable for the rest of this drive")
            self.soundmanager.play_autopilot_disabled()
            while True:
                self.throttle = -1000
                self.current_steering_angle = -1

        return -self.current_steering_angle - self.offset, self.throttle

    def process_line(self, line, image_to_show, color='blue', cv_color=(30, 30, 200)):
        for line_segment in line:
            for x1, y1, x2, y2 in line_segment:
                self.axs[3].plot([x1, x2], [y1, y2], color=color, linewidth=2)
                cv2.line(image_to_show, (x1, y1), (x2, y2), cv_color, 2)

    def plot_lines(self, normal_image, image, gray, edges, lines):

        if self.fig is None:
            # Start up the plotting
            self.fig = plt.figure(figsize=(10, 5))

            plt.ion()
            gs = gridspec.GridSpec(3, 2)  # define the grid
            self.axs = [self.fig.add_subplot(gs[0, 0]), self.fig.add_subplot(gs[0, 1]),
                        self.fig.add_subplot(gs[1, 0]), self.fig.add_subplot(gs[1, 1]),
                        self.fig.add_subplot(gs[2, :])]

            self.fig.canvas.manager.set_window_title('Autopilot system')
            self.soundmanager.play_autopilot_enabled()

        line_img = np.zeros_like(image)
        image_to_show = np.copy(image)

        if self.detected_left_line:
            self.process_line(self.left_line, image_to_show)
        elif self.history_left_lines:
            self.process_line(self.left_line, image_to_show, 'red', (255, 0, 0))
        if self.detected_right_line:
            self.process_line(self.right_line, image_to_show)
        elif self.history_right_lines:
            self.process_line(self.right_line, image_to_show, 'red', (255, 0, 0))
        else:
            print("No lines for plotting found")

        self.axs[0].cla()
        self.axs[0].imshow(gray, cmap='gray')
        self.axs[0].set_title('Gray filter')

        self.axs[1].cla()
        self.axs[1].imshow(edges, cmap='gray')
        self.axs[1].set_title('Edges filter')

        # Prepare line_img and default to the green color
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_img, (x1, y1), (x2, y2), (255, 255, 255), 1)

        self.axs[2].cla()
        self.axs[2].imshow(line_img, cmap='gray')
        self.axs[2].set_title('Detected lines')

        self.axs[3].cla()
        self.axs[3].imshow(image_to_show)
        self.axs[3].set_title('Computer view')

        self.axs[4].cla()
        self.axs[4].imshow(normal_image)
        steering_ratio = 100  # Adjust as needed

        # Define the start coordinates of the arrow
        x_start = image.shape[1] // 2
        y_start = image.shape[0]

        # Calculate the steering angle position on the arrow (inverted and shorter)
        arrow_length = 100  # Adjust the length of the arrow as needed
        x_stop = int(x_start - self.current_steering_angle * steering_ratio)
        y_stop = int(y_start - arrow_length)

        # Create the arrow
        self.axs[4].arrow(x_start, y_start, x_stop - x_start, y_stop - y_start, color='blue',
                          width=0.1, length_includes_head=True)

        self.axs[4].set_title('Steering')
        self.fig.tight_layout()
        self.fig.show()
        plt.show()
        plt.pause(0.001)
