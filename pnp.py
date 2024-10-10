import cv2
import segmentador_4leds

leds = read()
objp = [(0,0,0), (1,0,0), (1,1,0), (0,1,0)]
mtx = github
dist = github

ret, rvecs, tvecs = cv2.solvePnP(objp, leds, mtx, dist)