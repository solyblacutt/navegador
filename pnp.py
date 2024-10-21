import cv2
import segmentador_4leds




led_centers = []
objp = [(0,0,0), (1,0,0), (1,1,0), (0,1,0)]
#mtx = github
#dist = github

with open("led_centers.txt", "r") as f:
    # Leemos las líneas y guardamos las coordenadas en una lista
    for line in f:
        x, y = map(int, line.strip().split(","))
        led_centers.append((x, y))


print(led_centers)
#ret, rvecs, tvecs = cv2.solvePnP(objp, leds, mtx, dist)