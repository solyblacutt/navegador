import numpy as np
import cv2 as cv
import glob
import os

# ------------------ Parámetros de patrón ------------------
BOARD_SIZE = (8, 6)          # columnas x filas de esquinas internas
SQUARE_SIZE = 1.0            # unidad arbitraria (mm si querés); afecta tvec pero no el RMSE en px

# ------------------ Criterios de cornerSubPix ------------------
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

# ------------------ Función de estadísticas de reproyección ------------------
def reproj_stats(objpoints, imgpoints, rvecs, tvecs, K, dist):
    per_img_rmse = []
    per_img_n = []
    per_point_errors = []

    for i in range(len(objpoints)):
        objp = np.asarray(objpoints[i], np.float32)
        imgp = np.asarray(imgpoints[i], np.float32).reshape(-1, 2)  # (N,2)

        proj, _ = cv.projectPoints(objp, rvecs[i], tvecs[i], K, dist)
        proj = proj.reshape(-1, 2)

        e = np.linalg.norm(proj - imgp, axis=1)  # errores puntuales (px)
        per_point_errors.append(e)
        per_img_rmse.append(float(np.sqrt(np.mean(e**2))))
        per_img_n.append(len(e))

    all_e = np.concatenate(per_point_errors) if len(per_point_errors) else np.array([], dtype=float)
    rmse_global = float(np.sqrt(np.mean(all_e**2))) if all_e.size else np.nan  # ponderado por puntos
    rmse_prom_img = float(np.mean(per_img_rmse)) if per_img_rmse else np.nan   # promedio no ponderado
    p95 = float(np.percentile(all_e, 95)) if all_e.size else np.nan
    emax = float(np.max(all_e)) if all_e.size else np.nan

    return {
        "rmse_global": rmse_global,
        "rmse_promedio_imagen": rmse_prom_img,
        "rmse_por_imagen": per_img_rmse,
        "p95_error_puntual": p95,
        "max_error_puntual": emax,
        "n_puntos_por_imagen": per_img_n
    }

# ------------------ Puntos 3D del patrón (una vez) ------------------
# OJO: usa SQUARE_SIZE para que las tvec queden en mm si lo necesitás luego
objp_single = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
objp_single[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
objp_single *= SQUARE_SIZE

# ------------------ Acumuladores ------------------
objpoints = []     # puntos 3D reales (por imagen)
imgpoints = []     # puntos 2D detectados (por imagen)
used_filenames = []  # para mapear RMSE ↔ archivo

# ------------------ Cargar imágenes ------------------
images = glob.glob('calibrationCeluSol/*.jpg')
images.sort()
if not images:
    raise RuntimeError("No se encontraron imágenes en 'calibrationCeluSol/*.jpg'.")

for fname in images:
    img = cv.imread(fname)
    if img is None:
        print(f"Advertencia: no se pudo leer {fname}, se omite.")
        continue

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Buscar esquinas del tablero
    flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK
    ret, corners = cv.findChessboardCorners(gray, BOARD_SIZE, flags)

    if not ret:
        # Intento sin FAST_CHECK por si falló
        ret, corners = cv.findChessboardCorners(gray, BOARD_SIZE)

    if ret:
        # Refinamiento subpíxel
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp_single.copy())
        imgpoints.append(corners2)
        used_filenames.append(os.path.basename(fname))
    else:
        print(f"Tablero NO detectado en: {fname}")

if not objpoints:
    raise RuntimeError("No se detectó ningún tablero en las imágenes.")

# ------------------ Calibración ------------------
image_size = gray.shape[::-1]  # (w, h)
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, image_size, None, None
)

print("\n=== Resultados de calibración ===")
print('RMS (calibrateCamera ret): {:.6f} px'.format(ret))
print('Matriz de la cámara (K):\n', mtx)
print('Coeficientes de distorsión (dist):\n', dist.ravel())

# ------------------ Estadísticas de reproyección propias ------------------
stats = reproj_stats(objpoints, imgpoints, rvecs, tvecs, mtx, dist)

print("\n=== Estadísticas de reproyección (todas las imágenes) ===")
print("RMSE global (ponderado por puntos): {:.6f} px".format(stats['rmse_global']))
print("RMSE promedio por imagen          : {:.6f} px".format(stats['rmse_promedio_imagen']))
print("P95 error puntual                 : {:.6f} px".format(stats['p95_error_puntual']))
print("Máximo error puntual              : {:.6f} px".format(stats['max_error_puntual']))

# Top 5 peores imágenes (por RMSE)
rmse_list = stats['rmse_por_imagen']
order = np.argsort(rmse_list)[::-1]  # descendente
top = min(5, len(order))
print("\nPeores {} imágenes por RMSE:".format(top))
for rank in range(top):
    i = order[rank]
    fname = used_filenames[i] if i < len(used_filenames) else f"img_{i}"
    print("  #{:<2d} {:<30s}  RMSE={:.3f} px   Npts={}".format(
        rank+1, fname, rmse_list[i], stats['n_puntos_por_imagen'][i]
    ))

# ------------------ Undistorsión (opcional, demo) ------------------
demo_path = 'camara celu sol.jpg'
if os.path.exists(demo_path):
    img_demo = cv.imread(demo_path)
    if img_demo is not None:
        h, w = img_demo.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), cv.CV_32FC1)
        dst = cv.remap(img_demo, mapx, mapy, cv.INTER_LINEAR)

        x, y, w2, h2 = roi
        if w2 > 0 and h2 > 0:
            dst = dst[y:y + h2, x:x + w2]
        os.makedirs('trash', exist_ok=True)
        cv.imwrite('trash/calibresult.png', dst)
        print("\nImagen de undistorsión guardada en: trash/calibresult.png")
    else:
        print("\nAdvertencia: no se pudo leer la imagen de demo para undistorsión.")
else:
    print("\nNota: omito undistorsión de demo (no se encontró 'camara celu sol.jpg').")

# ------------------ Guardar parámetros ------------------
np.savez('B.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
print("\nParámetros guardados en B.npz")


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


""" CELU SOL IPHONE RPO
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

