import cv2


def filter_image(image, tuning_parameters):
    low_threshold, high_threshold, rho, theta, threshold, minlinelength, maxlinegap, kernel1, kernel2, \
        sigmax, slope_threshold = tuning_parameters

    # Crop the top & bottom portion of the image
    height, width = image.shape[:2]
    cropped_image = image[75:height - 30, :]

    # Apply Gaussian blur to the masked image
    blurred_image = cv2.GaussianBlur(cropped_image, (kernel1, kernel2), sigmax)

    # Convert the blurred image to grayscale
    gray = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    # Apply Hough Transform to detect lines
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=minlinelength,
                            maxLineGap=maxlinegap)

    if lines is not None:
        # Filter out near horizontal lines based on the scaled threshold
        slope_threshold_value = slope_threshold / 10.0  # Convert to a value between 0 and 1

        slopes_lines = [((y2 - y1) / (x2 - x1) if (x2 != x1) else float('inf'), line)
                        for line in lines for x1, y1, x2, y2 in line]

        lines = [line for slope, line in slopes_lines if abs(slope) > slope_threshold_value]

    return gray, edges, lines, cropped_image
