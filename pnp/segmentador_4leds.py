import cv2
import numpy as np

# Function to resize the image while maintaining the aspect ratio
def resize_image(image, max_width=800, max_height=600):
    height, width = image.shape[:2]
    scaling_factor = min(max_width / width, max_height / height)
    return cv2.resize(image, (int(width * scaling_factor), int(height * scaling_factor)))

# Load the image
image = cv2.imread('test4leds.jpeg')  # Replace with your image path

# Resize the image to fit the display window
resized_image = resize_image(image)

# Convert the image to grayscale
gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Threshold the image to detect bright objects (like LED lights)
_, thresh = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY)

# Find contours in the thresholded image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter the contours based on size to detect the LEDs
led_contours = []
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 50:  # Adjust based on your LED size
        led_contours.append(contour)

# Sort contours by position (optional, e.g., left to right)
led_contours = sorted(led_contours, key=lambda c: cv2.boundingRect(c)[0])

centers = []
with open("../led_centers.txt", "w") as file:
    for contour in led_contours[:4]:  # Limit to 4 LEDs
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        #txt que se borre y agrege los 4 centros
        file.write(f"{center[0]}, {center[1]}\n")
        print(center)
        radius = int(radius)
        cv2.circle(resized_image, center, radius, (0, 255, 0), 2)


        # Prepare the text for the coordinates (format: (x, y))
        coordinates_text = f"({center[0]}, {center[1]})"

        # Place the text above the circle (adjusting position as needed)
        cv2.putText(resized_image, coordinates_text, (center[0] - 40, center[1] - radius - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)



# Display the image with the detected LEDs highlighted
cv2.imshow('Detected LEDs', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



# (105, 295)
# (111, 367)
# (180, 367)
# (181, 307)