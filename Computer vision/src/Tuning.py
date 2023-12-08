import cv2
import numpy as np
import os

from src.ProcessImage import filter_image


def get_tuning_parameters_from_file(path):
    parameters = {}  # Create an empty dictionary to store the parameters.

    float_parameters_keys = ['theta']  # keys that represented float values
    # Open the text file in read mode.
    with open(path, 'r') as file:
        lines = file.readlines()

    # Go through each line in the file.
    for line in lines:
        try:
            # Remove whitespace from the start and end of the line, and split it where the '=' character is.
            # This separates the parameter name (key) and its value.
            key, value = line.strip().split('=')

            # Convert the value to an integer or float (if it's a float parameter key)
            # and store it in the dictionary under its parameter name.
            if key in float_parameters_keys:
                parameters[key] = float(value)
            else:
                parameters[key] = int(value)
        except ValueError:
            # If the line couldn't be split into exactly two parts or the value couldn't be converted to an integer,
            # print an error message.
            print(f"Ignored line '{line.strip()}' (cannot interpret as a key=value pair where value is an integer)")

    # Define a list of required parameters.
    required_params = ['low_threshold', 'high_threshold', 'rho', 'theta', 'threshold', 'minLineLength', 'maxLineGap',
                       'kernel1', 'kernel2', 'sigmax', 'slope_threshold']

    # Find out which required parameters were not found in the file.
    missing_params = [param for param in required_params if param not in parameters]
    if missing_params:
        # If there were missing parameters, print an error message.
        print(f"Error: Missing required parameters: {', '.join(missing_params)}")
    else:
        # If all required parameters were found, unpack them from the parameters dictionary to separate variables.
        low_threshold, high_threshold, rho, theta, threshold, minlinelength, maxlinegap, kernel1, kernel2, sigmax, slope_threshold = [
            parameters[param] for param
            in required_params]
        # Return those variables.
        return low_threshold, high_threshold, rho, theta, threshold, minlinelength, maxlinegap, kernel1, kernel2, sigmax, slope_threshold


def create_tuning_file(image, path):
    cv2.namedWindow('Finetuning', cv2.WINDOW_NORMAL)

    # Parameters for Canny filter
    # Any gradient value lower than "Low thresh" is considered not to be an edge.
    # The gradient values greater than the "High thresh" are considered edges.
    cv2.createTrackbar('Low thresh', 'Finetuning', 50, 255, lambda _: None)
    cv2.createTrackbar('High thresh', 'Finetuning', 150, 255, lambda _: None)

    # Parameters for Hough filter
    # 'rho' is the resolution of the parameter rho in pixels. 'theta' is the resolution of the 
    # parameter theta in radians (you express the range in degree which is then converted to radians in the Hough
    # transform function)
    cv2.createTrackbar('rho', 'Finetuning', 1, 3, lambda _: None)
    cv2.createTrackbar('theta', 'Finetuning', 1, 180,
                       lambda _: None)  # default value: 1 (representing 1 degree)

    # 'threshold' is the minimum number of intersections to "*detect*" a line
    cv2.createTrackbar('threshold', 'Finetuning', 100, 250, lambda _: None)

    # 'minLength' is the minimum length of the line that will be accepted.
    cv2.createTrackbar('minLength', 'Finetuning', 40, 100, lambda _: None)

    # 'maxGap' is the maximum gap between lines that will be allowed while recognizing it as a single line.
    cv2.createTrackbar('maxGap', 'Finetuning', 150, 250, lambda _: None)

    cv2.createTrackbar('kernel1', 'Finetuning', 5, 10, lambda _: None)
    cv2.createTrackbar('kernel2', 'Finetuning', 5, 10, lambda _: None)
    cv2.createTrackbar('sigmax', 'Finetuning', 0, 10, lambda _: None)
    cv2.createTrackbar('slope', 'Finetuning', 1, 10, lambda _: None)

    while True:
        low_threshold = cv2.getTrackbarPos('Low thresh', 'Finetuning')
        high_threshold = cv2.getTrackbarPos('High thresh', 'Finetuning')

        rho = int(cv2.getTrackbarPos('rho', 'Finetuning'))
        rho = max(1, rho)
        theta = cv2.getTrackbarPos('theta', 'Finetuning') * np.pi / 180  # theta in radian
        threshold = int(cv2.getTrackbarPos('threshold', 'Finetuning'))
        threshold = max(1, threshold)
        minLineLength = int(cv2.getTrackbarPos('minLength', 'Finetuning'))
        maxLineGap = int(cv2.getTrackbarPos('maxGap', 'Finetuning'))
        kernel1 = int(cv2.getTrackbarPos('kernel1', 'Finetuning'))
        kernel2 = int(cv2.getTrackbarPos('kernel2', 'Finetuning'))
        kernel1 = max(1, kernel1)  # Ensure it's at least 1
        kernel2 = max(1, kernel2)  # Ensure it's at least 1
        kernel1 = kernel1 + 1 if kernel1 % 2 == 0 else kernel1  # Ensure it's odd
        kernel2 = kernel2 + 1 if kernel2 % 2 == 0 else kernel2  # Ensure it's odd

        sigmax = int(cv2.getTrackbarPos('sigmax', 'Finetuning'))
        slope_threshold = int(cv2.getTrackbarPos('slope', 'Finetuning'))

        tuning_parameters = low_threshold, high_threshold, rho, theta, threshold, minLineLength, maxLineGap, kernel1, kernel2, sigmax, slope_threshold

        # Let's filter the image so we get a live preview
        gray, edges, lines, cropped_image = filter_image(image, tuning_parameters)

        # Drawing lines on the image
        line_img = np.zeros_like(gray)
        image_to_show = np.copy(gray)
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_img, (x1, y1), (x2, y2), (255, 255, 255), 1)
                    cv2.line(image_to_show, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Ensure all images are the same size
        edges = cv2.resize(edges, (gray.shape[1], gray.shape[0]))
        line_img = cv2.resize(line_img, (gray.shape[1], gray.shape[0]))

        # Concatenate images horizontally
        combined_image = cv2.hconcat([edges, line_img, image_to_show])

        # Display the concatenated image
        cv2.imshow('Finetuning', combined_image)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyWindow('Finetuning')

    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Write the parameters to the file.
    with open(path, 'w') as file:
        print("Saving parameters to file")
        file.write(f'low_threshold={low_threshold}\n')
        file.write(f'high_threshold={high_threshold}\n')
        file.write(f'rho={rho}\n')
        file.write(f'theta={theta}\n')
        file.write(f'threshold={threshold}\n')
        file.write(f'minLineLength={minLineLength}\n')
        file.write(f'maxLineGap={maxLineGap}\n')
        file.write(f'kernel1={kernel1}\n')
        file.write(f'kernel2={kernel2}\n')
        file.write(f'sigmax={sigmax}\n')
        file.write(f'slope_threshold={slope_threshold}\n')
