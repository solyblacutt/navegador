import numpy as np
import cv2

# ---------- Reproyección ---------- error de ver ptos 3D en 2D y compararlos con imagen 2D --- dominio 2D
def rmse_reproyeccion(objp, imgp, rvec, tvec, K, dist):
    objp = np.asarray(objp, np.float32)
    imgp = np.asarray(imgp, np.float32)
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    proj = proj.reshape(-1,2)
    e = np.linalg.norm(proj - imgp, axis=1)
    return float(np.sqrt((e**2).mean())), e  # RMSE, errores individuales

# ---------- FRE / TRE ----------
# FRE > uso ptos anato y pntos modelo del REGISTRO
def fre(pts_mundo, pts_modelo, R, t):
    """
    pts_mundo: Nx3 medidos con tu punzón (sistema "mundo/paciente")
    pts_modelo: Nx3 correspondientes en el modelo (mm)
    R,t: transform que lleva mundo -> modelo (o alinear conforme a tu convención)
    """
    A = np.asarray(pts_mundo, float)
    B = np.asarray(pts_modelo, float)
    pred = (R @ A.T).T + t
    e = np.linalg.norm(pred - B, axis=1)
    return float(np.sqrt((e**2).mean())), e


# TRE > uso R y t generados en FRE con punto tomado de anato y dibujo pto en el modelo
def tre(pts_mundo_test, pts_modelo_test, R, t):
    return fre(pts_mundo_test, pts_modelo_test, R, t)

# ---------- Dice (2D o 3D) ----------
def dice_coefficient(mask_pred, mask_gt):
    """
    mask_*: arrays booleanos (H×W) o (Z×Y×X).
    """
    A = mask_pred.astype(bool); B = mask_gt.astype(bool)
    inter = np.logical_and(A,B).sum()
    s = A.sum() + B.sum()
    if s == 0:
        return 1.0
    return 2.0*inter/s


#-----------------------------

puntos_mundo = [[-105.435,73.532,-153.527],
[-113.563,46.563,-138.513],
[-74.543,73.356,-236.942]]

puntos_modelo = [[-99.01024627685547,62.421653747558594,-165.8543701171875],
[-106.19417572021484,39.02788543701172,-166.59251403808594],
[-66.40510559082031,58.021236419677734,-224.09645080566406]]

rvecs = np.array([1.72797138, 1.81628202, 2.26308698])
# tvecs = [[-20.38184396],
#  [-25.25523136],
#  [ 47.66751124]]



def rigid_fit(A, B):
    A = np.asarray(A, float); B = np.asarray(B, float)
    muA, muB = A.mean(0), B.mean(0)
    A0, B0 = A - muA, B - muB
    U, S, Vt = np.linalg.svd(A0.T @ B0)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:  # asegurar rotación propia
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = muB - R @ muA
    return R, t

R_mc, t_mc = rigid_fit(puntos_mundo, puntos_modelo)  # modelo ← cámara (o el marco que uses)




# Cargar el archivo .npz
data = np.load('B.npz')
# Acceder a los arrays guardados
mtx = data['mtx']
dist = data['dist']
#rvecs = data['rvecs']
tvecs = np.array(data['tvecs'])

R, _ = cv2.Rodrigues(rvecs)

fre_inicial = fre(puntos_mundo,puntos_modelo,R_mc,t_mc)
print(fre_inicial)

# transformo toma de puntos, mismo R y t del registro, comparo punt ocon modelo
punto_mundo_verif = [[-95.241,57.184,-160.194]] #punto tomado posterior al registro

punto_mundo_verif2 = [[-70.31375885009766, 46.940242767333984, -171.58868408203125]] # (20.896832755079902, array([20.89683276]))

punto_modelo_verif = [[-70.31375885009766, 46.940242767333984, -171.58868408203125]] # preparo otro pto del modelo para esto

tre_inicial = tre(punto_mundo_verif2, punto_modelo_verif,R_mc, t_mc)
print(tre_inicial)
