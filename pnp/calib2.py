import numpy as np
import cv2 as cv
import glob

# Criterios de terminación para cornerSubPix
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Puntos del patrón 3D (en el mundo real)
objp = np.zeros((6 * 8, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

# Listas para almacenar puntos 3D y 2D
objpoints = []  # puntos 3D reales
imgpoints = []  # puntos 2D detectados en las imágenes

# Cargar todas las imágenes de calibración
images = glob.glob('calibration/*.png')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


    # Buscar esquinas del patrón de tablero de ajedrez 8x6
    ret, corners = cv.findChessboardCorners(gray, (8, 6), None)

    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Dibujar las esquinas detectadas
        # cv.drawChessboardCorners(img, (8, 6), corners2, ret)
        # cv.imshow('Esquinas detectadas', img)
        # cv.waitKey(500)

# Calibración de la cámara
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print('Matriz de la cámara (mtx):\n', mtx)
print('Coeficientes de distorsión (dist):\n', dist)
print('Vectores de rotación (rvecs):\n', rvecs)

# Leer imagen nueva para refinar calibración
img = cv.imread('../IMG_4204.png')
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

print('Matriz nueva de la cámara (newcameramtx):\n', newcameramtx)

# Undistorsionar usando remap (método 2)
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# Recortar ROI
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv.imwrite('../calibresult.png', dst)

# Calcular error de reproyección
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

print("Error total de reproyección: {:.6f}".format(mean_error / len(objpoints)))

# Guardar resultados para uso posterior en estimación de pose
np.savez('../B.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)


"""
Matriz de la cámara (mtx):
 [[2.99245904e+03 0.00000000e+00 1.48942745e+03]
 [0.00000000e+00 2.97952349e+03 2.00395063e+03]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
Coeficientes de distorsión (dist):
 [[ 2.43302187e-01 -1.29527262e+00 -2.50888650e-03 -2.07908157e-03
   2.41003838e+00]]
Vectores de rotación (rvecs):
 (array([[0.39086223],
       [0.02025785],
       [1.56472827]]), array([[0.07966633],
       [0.34604445],
       [1.63510694]]), array([[ 0.03070445],
       [-0.15177614],
       [-1.54250506]]), array([[-0.12304991],
       [-0.24930433],
       [ 1.54294704]]), array([[ 0.171478  ],
       [-0.37199603],
       [ 1.50612845]]), array([[-0.47012702],
       [ 0.31259616],
       [ 1.45226359]]), array([[ 0.08000001],
       [-0.26525256],
       [-1.52666145]]), array([[-0.00318024],
       [-0.09683786],
       [-1.53983298]]), array([[0.14642944],
       [0.27421689],
       [1.63143249]]), array([[ 0.00501963],
       [ 0.27837337],
       [-1.42719965]]), array([[-0.01195879],
       [-0.17719996],
       [ 1.59930323]]), array([[-0.17320817],
       [-0.11786455],
       [ 1.58683304]]), array([[-0.33678609],
       [ 0.13577626],
       [ 1.50892285]]), array([[-0.37944523],
       [-0.28411717],
       [-1.48522738]]), array([[-0.45930997],
       [ 0.13410785],
       [ 1.53387876]]), array([[-0.32898226],
       [-0.14570631],
       [ 1.55899101]]), array([[ 0.20289672],
       [-0.07631234],
       [-1.45995945]]), array([[0.085275  ],
       [0.18418446],
       [1.56606341]]), array([[ 0.28803808],
       [ 0.30386106],
       [-1.47398626]]), array([[-0.35852827],
       [-0.28714814],
       [ 1.58700202]]), array([[-0.3432819 ],
       [-0.57016227],
       [-1.48223223]]), array([[0.09822911],
       [0.25089104],
       [1.56689457]]))
Matriz nueva de la cámara (newcameramtx):
 [[3.04418637e+03 0.00000000e+00 1.48538816e+03]
 [0.00000000e+00 3.12958031e+03 1.99984315e+03]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
Error total de reproyección: 0.223445
"""
