import numpy as np
import cv2 as cv
import glob
from matplotlib import pyplot as plt 

# Load previously saved data
with np.load('B.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype(int))
    img = cv.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (0,0,255), 5)
    return img

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
 
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

print('sapo')
for fname in glob.glob('calibration/*.png'):
    img = cv.imread(fname)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (8,6),None)
    print(ret)
    if ret == True:
        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
 
        # Find the rotation and translation vectors.
        ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
 
        # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
 
        img = draw(img,corners2,imgpts)

        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.show()

        #cv.imshow('img',img)
        k = cv.waitKey(0) & 0xFF
        if k == ord('s'):
            print('sapo2')
            cv.imwrite(fname[:6]+'.png', img)
 
#cv.destroyAllWindows()