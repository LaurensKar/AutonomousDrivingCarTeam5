import os.path
import pickle
import socketio
import eventlet
import base64
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------- Settings ---------------------------------------------

Finetune = True  # Specify if you want to finetune the filter again
plot_shown = False  # Choose if you want to show a summary plot or not

FinetuningFile = 'finetuning.pic'  # Those are the parameters that the filters will use


# -------------------------------------------------- Functions ---------------------------------------------

# This function is used to easily finetune our parameters.
def finetuning(image):
    if Finetune:
        if os.path.isfile(FinetuningFile):
            with open(FinetuningFile, 'rb') as f:
                parameters = pickle.load(f)
                low_threshold, high_threshold, rho, theta, threshold, minLineLength, maxLineGap = parameters
        else:
            cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
            cv2.createTrackbar('Low threshold', 'Image', 0, 500, lambda _: None)
            cv2.createTrackbar('High threshold', 'Image', 0, 500, lambda _: None)

            while True:
                low_threshold = cv2.getTrackbarPos('Low threshold', 'Image')
                high_threshold = cv2.getTrackbarPos('High threshold', 'Image')
                edges = cv2.Canny(image, low_threshold, high_threshold)
                cv2.imshow('Image', edges)

                if cv2.waitKey(1) == ord('q'):
                    break

            cv2.destroyWindow('Image')

            cv2.namedWindow('Hough Lines', cv2.WINDOW_NORMAL)
            cv2.createTrackbar('rho', 'Hough Lines', 1, 10, lambda _: None)
            cv2.createTrackbar('theta', 'Hough Lines', 1, 360, lambda _: None)
            cv2.createTrackbar('threshold', 'Hough Lines', 1, 500, lambda _: None)
            cv2.createTrackbar('minLineLength', 'Hough Lines', 1, 100, lambda _: None)
            cv2.createTrackbar('maxLineGap', 'Hough Lines', 1, 100, lambda _: None)

            while True:
                rho = cv2.getTrackbarPos('rho', 'Hough Lines')
                theta = cv2.getTrackbarPos('theta', 'Hough Lines') * np.pi / 180
                threshold = cv2.getTrackbarPos('threshold', 'Hough Lines')
                minLineLength = cv2.getTrackbarPos('minLineLength', 'Hough Lines')
                maxLineGap = cv2.getTrackbarPos('maxLineGap', 'Hough Lines')

                # lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength, maxLineGap)

                lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=10,
                                        maxLineGap=250)

                # drawing lines on the image
                if lines is not None:
                    line_img = np.zeros_like(image)
                    for line in lines:
                        for x1, y1, x2, y2 in line:
                            cv2.line(line_img, (x1, y1), (x2, y2), (255, 255, 255), 1)
                    cv2.imshow('Hough Lines', line_img)  # fixed the args to cv2.imshow

                image_to_show = np.copy(image)
                if lines is not None:
                    for line in lines:
                        for x1, y1, x2, y2 in line:
                            cv2.line(image_to_show, (x1, y1), (x2, y2), (0, 255, 0), 3)

                cv2.imshow('Hough Lines', image_to_show)  # fix: use image as input, not lines

                if cv2.waitKey(1) == ord('q'):
                    break

            cv2.destroyWindow('Hough Lines')

            with open(FinetuningFile, 'wb') as f:
                pickle.dump((low_threshold, high_threshold, rho, theta, threshold, minLineLength, maxLineGap), f)

    else:
        if os.path.isfile(FinetuningFile):
            with open(FinetuningFile, 'rb') as f:
                parameters = pickle.load(f)
            low_threshold, high_threshold, rho, theta, threshold, minLineLength, maxLineGap = parameters
        else:
            print("No finetuning file found. Set finetuning to TRUE. Sending default values")
            low_threshold = 50
            high_threshold = 150
            rho = 1
            theta = np.pi / 180
            threshold = 15
            minLineLength = 10
            maxLineGap = 10

    edges = cv2.Canny(image, low_threshold, high_threshold)
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength, maxLineGap)

    return edges, lines


def calculate_steering_angle(image):
    global plot_shown
    fig, axs = plt.subplots(3)

    # We grayscale our images because of simplicity, computational efficiency and edge strength.
    # Having one channel instead of 3 (red, green and blue) makes it easier to process and analyze the image. It is also
    # computationally more efficient. Lastly grayscaling makes sure that all pixels have the same intensity level, this is helpful
    # because edges are defined by abrupt changes in intensity or color, so by removing color it only focuses on intensity level.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges, lines = finetuning(gray)

    axs[0].imshow(gray, cmap='gray')
    axs[0].set_title('Gray Image')

    axs[1].imshow(edges, cmap='gray')
    axs[1].set_title('Edges')
    # drawing lines on the image
    if lines is not None:
        line_img = np.zeros_like(image)
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_img, (x1, y1), (x2, y2), (255, 255, 255), 1)
        axs[2].imshow(line_img, cmap='gray')
        axs[2].set_title('Lines')

     # When you visualize a road going off into the distance the 2 edges of the road go towards eachother.
    # This is what we will use to calculate the driving angle that our car needs to steer towards.
    # First we find the slope of the edges of the racetrack
    slopes = [(y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0 for line in lines for x1, y1, x2, y2 in line]
    # Convert the slopes into angles measured in radians which we later use to make calculations
    radian_angles = [np.arctan(x) for x in slopes]

    # Of course our filters and line transform aren't perfect and we will end up with quite some horizontal lines too.
    # We know however that the edges of our racetrack (the only lines we want) will almost never be horizontal and so we filter out all
    # lines that are less than 15 degrees.
    threshold_angle_degrees = 15

    # Convert the threshold angle to radians so we can calculate with it
    threshold_angle_radians = np.deg2rad(threshold_angle_degrees)

    # Remove near horizontal lines based on the constant threshold
    radian_angles = [x for x in radian_angles if abs(x) > threshold_angle_radians]

    # We now figure out the steering angle by calculating the mean of the slopes of the lines (hopefully only the edges of our racetrack)
    # that still remain. This angle is where we want to steer our car towards.
    # Take the mean of the angles which is our steering direction
    steering_angle_radian = np.mean(radian_angles)

    # Convert to degrees
    steering_angle_degree = np.rad2deg(steering_angle_radian)

    if not plot_shown:
        fig.tight_layout()  # adjusting layout to avoid overlaps
        plt.show()  # show the plot
        plot_shown = True  # After showing the plot, set the flag to True

    return steering_angle_degree

# Send the steering angle and the throttle to Udacity.
def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    })


# -------------------------------------------------- Connection ---------------------------------------------

sio = socketio.Server()


@sio.on('connect')  # Connect to the game
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)

# This funtion processes telemetry data received from the simulator and sends control commands back to the simulator
@sio.on('telemetry')
def telemetry(sid, data):
    print('Sending control')
    image_str = data["image"]
    image = Image.open(BytesIO(base64.b64decode(image_str)))

    angle = calculate_steering_angle(np.array(image))
    send_control(angle, 50)


app = socketio.WSGIApp(sio)
eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
