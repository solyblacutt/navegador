import numpy as np
import cv2

# ---------- Reproyección ----------
def rmse_reproyeccion(objp, imgp, rvec, tvec, K, dist):
    objp = np.asarray(objp, np.float32)
    imgp = np.asarray(imgp, np.float32)
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    proj = proj.reshape(-1,2)
    e = np.linalg.norm(proj - imgp, axis=1)
    return float(np.sqrt((e**2).mean())), e  # RMSE, errores individuales

# ---------- FRE / TRE ----------
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

def tre(pts_mundo_test, pts_modelo_test, R, t):
    return fre(pts_mundo_test, pts_modelo_test, R, t)

# ---------- Precisión (repeatability) del tip ----------
def precision_tip(poses_tip_3D):
    """
    poses_tip_3D: Mx3 (múltiples mediciones del tip sobre un objetivo estático).
    Devuelve std_xyz (mm) y std_radial (mm).
    """
    P = np.asarray(poses_tip_3D, float)
    mu = P.mean(0)
    dif = P - mu
    std_xyz = P.std(0)
    std_radial = float(np.sqrt((dif**2).sum(axis=1).mean()))
    return std_xyz, std_radial

# ---------- Detección: precisión/recall/exactitud ----------
def metricas_deteccion(pred_pts, gt_pts, tol_px=5.0):
    """
    pred_pts, gt_pts: listas de (x,y). Match greedy por distancia < tol.
    """
    pred = np.array(pred_pts, float) if len(pred_pts)>0 else np.zeros((0,2))
    gt   = np.array(gt_pts, float)   if len(gt_pts)>0 else np.zeros((0,2))
    if len(gt)==0 and len(pred)==0:
        return dict(precision=1.0, recall=1.0, accuracy=1.0, TP=0, FP=0, FN=0)
    used_pred = set(); TP=0
    for g in gt:
        d = np.linalg.norm(pred - g, axis=1) if len(pred)>0 else np.array([])
        if len(d)==0:
            continue
        j = int(np.argmin(d))
        if d[j] <= tol_px and j not in used_pred:
            TP += 1; used_pred.add(j)
    FP = len(pred) - len(used_pred)
    FN = len(gt) - TP
    precision = TP / (TP+FP) if (TP+FP)>0 else 0.0
    recall    = TP / (TP+FN) if (TP+FN)>0 else 0.0
    # "accuracy" aquí es sobre positivos esperados (aprox útil para seguimiento)
    denom = TP+FP+FN
    accuracy = TP/denom if denom>0 else 0.0
    return dict(precision=precision, recall=recall, accuracy=accuracy, TP=TP, FP=FP, FN=FN)

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
