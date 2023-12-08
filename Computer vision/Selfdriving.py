import os.path
import socketio
import eventlet
import base64
from PIL import Image
from io import BytesIO
import numpy as np
import configparser

from src.ControlPanel import create_and_show_control_panel
from src.Logger import setup_logger
from src.ProcessImage import filter_image
from src.AutosteerSystem import Autosteer
from src.SoundManager import SoundManager
from src.Tuning import create_tuning_file, get_tuning_parameters_from_file


# -------------------------------------------------- Function ---------------------------------------------

def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    })


# -------------------------------------------------- The code ---------------------------------------------

sio = socketio.Server()
soundmanager = SoundManager()
logger = setup_logger()
config = configparser.ConfigParser()
autosteersystem = Autosteer()


@sio.on('connect')  # Connect to the game
def connect(_, __):
    print('Connected to application')
    logger.info('Connected to the application')

    config.read('config/config.ini')
    create_and_show_control_panel()
    soundmanager.play_parking_brake_disabled()


@sio.on('telemetry')  # Get current state of the car
def telemetry(_, data):
    # 1. Get the fpv and put it in an image
    image_str = data["image"]
    image = Image.open(BytesIO(base64.b64decode(image_str)))
    numpy_image = np.array(image)

    # 2. Get the finetuning parameters
    file_path = config.get('DEFAULT', 'FilePath')
    if not os.path.isfile(file_path):
        print("No finetuning file found. Let's create one ourselves!")
        logger.warning("No finetuning file found, creating one ourselves")
        create_tuning_file(numpy_image, file_path)  # Create tuning file
    tuning_settings = get_tuning_parameters_from_file(file_path)

    # 3. Preprocess the image
    gray, edges, lines, cropped_image = filter_image(numpy_image, tuning_settings)

    # 4. Detect slopes and split left and right lane
    autosteersystem.detect_left_and_right_lines(lines)

    # 5. Save the line into memory and update the throttle (e.g slow down when it loses the lines)
    autosteersystem.save_lines_into_memory_and_update_throttle()

    # 6. Calculate the steering angle
    angle, throttle = autosteersystem.steering_calculation()

    # 7. Send the controls to the car
    send_control(angle, throttle)

    # 8. Show a live view of what the computer sees
    autosteersystem.plot_lines(numpy_image, cropped_image, gray, edges, lines)


app = socketio.WSGIApp(sio)
eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
