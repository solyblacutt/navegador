import cv2
import numpy as np


led_centers = []
objp = [(0,0,0), (1,0,0), (1,1,0), (0,1,0)]


with open("../led_centers.txt", "r") as f:
    # Lgit branch -aeemos las líneas y guardamos las coordenadas en una lista
    for line in f:
        x, y = map(int, line.strip().split(","))
        led_centers.append((x, y))

print(led_centers)

mtx = [(2.99245904e+03, 0.00000000e+00, 1.48942745e+03),
       (0.00000000e+00, 2.97952349e+03, 2.00395063e+03),
       (0.00000000e+00, 0.00000000e+00, 1.00000000e+00)]

dist = [ 2.43302187e-01, -1.29527262e+00, -2.50888650e-03, -2.07908157e-03,
   2.41003838e+00]

ret, rvecs, tvecs = cv2.solvePnP(objp, led_centers, mtx, dist)