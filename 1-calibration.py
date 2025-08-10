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
images = glob.glob('calibrationCeluSol/*.jpg')

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
img = cv.imread('camara celu sol.jpg')
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

print('Matriz nueva de la cámara (newcameramtx):\n', newcameramtx)

# Undistorsionar usando remap (método 2)
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# Recortar ROI
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv.imwrite('trash/calibresult.png', dst)

# Calcular error de reproyección
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

print("Error total de reproyección: {:.6f}".format(mean_error / len(objpoints)))

# Guardar resultados para uso posterior en estimación de pose
np.savez('B.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)


"""　Matriz de camara del celu:
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
#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------

"""
MATRIZ CAMARA COMPU SOL

Matriz de la cámara (mtx):
 [[1.23994321e+03 0.00000000e+00 9.42066048e+02]
 [0.00000000e+00 1.24162269e+03 5.16545687e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
Coeficientes de distorsión (dist):
 [[ 0.02489004  0.12455246 -0.01055148  0.00068239  0.14485304]]
Vectores de rotación (rvecs):
 (array([[ 0.07853689],
       [ 0.00742668],
       [-0.05599506]]), array([[-0.38448845],
       [ 0.26394945],
       [-0.0632318 ]]), array([[-0.02468243],
       [ 0.08597906],
       [ 0.25498821]]), array([[ 0.03911515],
       [ 0.14873545],
       [-0.04126733]]), array([[-0.01673839],
       [ 0.19299242],
       [-0.78502796]]), array([[ 0.01025896],
       [ 0.2659633 ],
       [-1.33033101]]), array([[ 0.34780951],
       [ 0.11924022],
       [-0.65412479]]), array([[-0.12835463],
       [ 0.14419748],
       [-0.40230986]]), array([[-0.1402249 ],
       [ 0.22767007],
       [-0.99483065]]), array([[-0.04702009],
       [ 0.12979066],
       [-0.21424732]]), array([[-0.05420355],
       [ 0.0972936 ],
       [ 0.17299337]]), array([[-0.00603707],
       [ 0.12435625],
       [ 0.17276984]]), array([[ 0.0096517 ],
       [ 0.19545842],
       [-0.57823833]]), array([[-0.10796241],
       [ 0.18977246],
       [-1.11734206]]), array([[-0.20478683],
       [ 0.24603542],
       [-1.2142609 ]]), array([[ 0.04097628],
       [ 0.14094598],
       [-0.54650078]]), array([[-0.05401518],
       [ 0.14351113],
       [ 0.04777998]]), array([[0.2327201 ],
       [0.15626209],
       [0.57565812]]), array([[0.14540799],
       [0.21492446],
       [0.27077973]]), array([[0.1615952 ],
       [0.13001797],
       [0.73990138]]), array([[0.13896499],
       [0.26953037],
       [0.37694347]]), array([[ 0.19010657],
       [-0.01184551],
       [ 1.21493436]]))
Matriz nueva de la cámara (newcameramtx):
 [[1.32460873e+03 0.00000000e+00 9.46875384e+02]
 [0.00000000e+00 1.25231889e+03 5.10200098e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
Error total de reproyección: 0.113960

"""


#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------


"""
Matriz de la cámara (mtx):
 [[626.61530077   0.         347.84841745]
 [  0.         628.50695348 367.25298278]
 [  0.           0.           1.        ]]
 
Coeficientes de distorsión (dist):
 [[-0.13502325  0.15097997 -0.01129146 -0.01133406 -0.05517456]]
 
Vectores de rotación (rvecs):
 (array([[-0.24698016],
       [ 0.14380529],
       [-1.29844009]]), array([[-0.34034595],
       [ 0.28710651],
       [-1.37307783]]), array([[-0.21339454],
       [ 0.0450549 ],
       [-1.28280933]]), array([[ 0.04568448],
       [-0.20591427],
       [-1.30972848]]), array([[ 0.23910528],
       [-0.14606182],
       [-1.38790998]]), array([[ 0.22002383],
       [ 0.27616275],
       [-1.45281864]]), array([[-0.52714487],
       [-0.35980056],
       [-1.39396319]]), array([[0.23546784],
       [0.61702106],
       [1.63007912]]), array([[0.2886269 ],
       [0.65421214],
       [1.66318398]]), array([[-0.31932604],
       [ 0.24547101],
       [-1.20602868]]), array([[-0.09156477],
       [ 0.17276534],
       [-1.00930329]]), array([[-0.09133229],
       [ 0.56472067],
       [-1.20066807]]), array([[ 0.10051181],
       [ 0.26257984],
       [-1.11657825]]), array([[ 0.16994653],
       [-0.03147107],
       [-1.20987054]]), array([[ 0.13300191],
       [-0.02698598],
       [-1.22619655]]), array([[-0.16441476],
       [ 0.1279065 ],
       [-1.47159982]]), array([[0.23367097],
       [0.45184534],
       [1.54843424]]), array([[0.36789734],
       [0.58172892],
       [1.47496445]]), array([[-0.62503018],
       [ 0.03457515],
       [ 0.06038582]]), array([[-0.34398645],
       [ 0.27248117],
       [ 0.32005011]]), array([[-0.50343965],
       [ 0.38644431],
       [ 0.25077927]]), array([[4.30720339e-04],
       [3.46783335e-01],
       [4.39776056e-01]]), array([[0.06639   ],
       [0.14553641],
       [0.13909829]]), array([[0.39123874],
       [0.0244238 ],
       [0.17082561]]), array([[-0.20814761],
       [-0.27899492],
       [ 0.27387239]]), array([[-0.47443392],
       [-0.05614368],
       [ 0.1007118 ]]), array([[-0.46367623],
       [-0.05578899],
       [ 0.09938547]]), array([[-0.18141576],
       [ 0.25501781],
       [ 0.2124853 ]]), array([[-0.37479903],
       [ 0.45496198],
       [ 0.19847022]]), array([[-0.11350895],
       [ 0.55377088],
       [ 0.36409122]]), array([[-0.54207097],
       [-0.08965923],
       [ 0.1106371 ]]), array([[-0.51746405],
       [ 0.58636054],
       [ 0.15675467]]), array([[0.10093801],
       [0.44291013],
       [0.36144589]]), array([[-0.01933763],
       [ 0.28496659],
       [ 0.31660503]]), array([[-0.64289184],
       [ 0.28968613],
       [ 0.18791298]]), array([[-0.13589051],
       [ 0.31180655],
       [ 0.2954754 ]]), array([[0.60969538],
       [0.08302902],
       [0.2302433 ]]), array([[-0.61067824],
       [ 0.19285256],
       [ 0.28476647]]))

Matriz nueva de la cámara (newcameramtx):
 [[591.80693821   0.         337.44814225]
 [  0.         592.61559322 354.94961427]
 [  0.           0.           1.        ]]
Error total de reproyección: 0.205710
"""