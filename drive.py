import socketio
import eventlet
import base64
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------- Settings ---------------------------------------------

Finetune = False
plot_shown = False

FinetuningFile = 'finetune.pic'


# -------------------------------------------------- Functions ---------------------------------------------

def calculate_steering_angle(image):
    global plot_shown
    fig, axs = plt.subplots(3)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    axs[0].imshow(gray, cmap='gray')
    axs[0].set_title('Gray Image')

    edges = cv2.Canny(gray, 100, 200)   # increased thresholds for edge detection
    axs[1].imshow(edges, cmap='gray')
    axs[1].set_title('Edges')

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=60, maxLineGap=300)   

    # drawing lines on the image
    if lines is not None:
        line_img = np.zeros_like(image)
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_img, (x1, y1), (x2, y2), (255, 255, 255), 1)
        axs[2].imshow(line_img, cmap='gray')
        axs[2].set_title('Lines')

    # Find the slope of the line
    # Since the slope of the line will give us an idea of the angle of the road
    slopes = [(y2 - y1)/(x2 - x1) if (x2-x1) != 0 else 0 for line in lines for x1,y1,x2,y2 in line]
    radian_angles = [np.arctan(x) for x in slopes]

    # Remove near horizontal lines based on a threshold
    radian_angles = [x for x in radian_angles if abs(x) > np.pi/12]

    # Take the mean of the angles which is our steering direction
    steering_angle_radian = np.mean(radian_angles)

    # Convert to degrees
    steering_angle_degree = np.rad2deg(steering_angle_radian)

    if not plot_shown:
        fig.tight_layout()  # adjusting layout to avoid overlaps
        plt.show()  # show the plot

        # After showing the plot, set the flag to True
        plot_shown = True
        
    return steering_angle_degree


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


@sio.on('telemetry')
def telemetry(sid, data):
    print('Sending control')
    image_str = data["image"]
    image = Image.open(BytesIO(base64.b64decode(image_str)))
    
    angle = calculate_steering_angle(np.array(image))
    send_control(angle, 50)
    
    
# -------------------------------------------------- Code ---------------------------------------------

app = socketio.WSGIApp(sio)
eventlet.wsgi.server(eventlet.listen(('', 4567)), app)






