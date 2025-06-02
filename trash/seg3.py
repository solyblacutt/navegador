import cv2
import numpy as np

img = cv2.imread('../pnp/test4leds.jpeg', cv2.IMREAD_COLOR)

# Get the dimensions of the image
height, width = img.shape[:2]

# Define the maximum width and height for display (e.g., 800x600)
max_width = 800
max_height = 600

# Calculate the scaling factor to maintain the aspect ratio
scaling_factor = min(max_width / width, max_height / height)

# Resize the image using the scaling factor
resized_image = cv2.resize(img, (int(width * scaling_factor), int(height * scaling_factor)))

# It converts the BGR color space of image to HSV color space
hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

# Threshold of blue in HSV space
lower_blue = np.array([60, 35, 140])
upper_blue = np.array([180, 255, 255])

# preparing the mask to overlay
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# The black region in the mask has the value of 0,
# so when multiplied with original image removes all non-blue regions
result = cv2.bitwise_and(resized_image, resized_image, mask=mask)

cv2.imshow('img', resized_image)
cv2.imshow('mask', mask)
cv2.imshow('result', result)

cv2.waitKey(0)

cv2.destroyAllWindows()