import numpy as np
import cv2 as cv
import glob
from matplotlib import pyplot as plt
 
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
 
# Arrays to store object points and image points from all the images.
objpoints = [(0,0,0), (1,0,0)] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
 
images = glob.glob('calibration/*.png')

for fname in images:
    
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    # 7x6 por el grid de la foto, el patron que tiene que encontrar
    # ret == True si encuentra el patron
    # puedo usar un patron circular, ventaja: requiere menos imagenes
    ret, corners = cv.findChessboardCorners(gray, (8,6), None)
    print(ret)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        # mejoro la precision con cornerSubPix()
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
 
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.show()

        # no usamos cv.imshow() porque desde VScode no deja correrlo bien, con otro IDE con GUI deberia
        # Draw and display the corners
        #cv.drawChessboardCorners(img, (8,6), corners2, ret)
        #cv.imshow('img', img)
        #cv.waitKey(500)
 
#cv.destroyAllWindows()


ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print('mtx: \n')
print(mtx)
print('dist: \n')
print(dist)
print('rvecs: \n')
print(rvecs)
# refino la matriz
# left12 es una imagen nueva, no del set que paso
img = cv.imread('IMG_4204.png')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

print('newcameramtx: \n')
print(newcameramtx)

# --------- 2 formas de undistorsionar ------------ probar cual da un mejor resultado
"""
# 1) total error: 0.2234451935929253
# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
 
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)
print('dist: \n')
print(dst)

"""
# 2) total error: 0.2234451935929253
# undistort
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
 
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst) #guardo la img con el roi de interes


# estimacion del error. muentras mas cerca de 0 este, mejor
# no depende del metodo para undistort
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
 
print( "total error: {}".format(mean_error/len(objpoints)) )

#guardo todo en un archivo para usarlo en pose-estimation
np.savez('B.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)