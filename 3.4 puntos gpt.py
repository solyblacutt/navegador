# 3-tomo-3-puntos_droidcam_roja_punzon_lite_v2.py
import cv2
import numpy as np
import math

# ============== Cámara (DroidCam) ==============
IP, PORT = "192.168.0.91", "4747"   # cambia por lo de tu app
URL = f"http://{IP}:{PORT}/video"

# --- Calibración de cámara ---
# Cargar el archivo .npz
data = np.load('B.npz')
# Acceder a los arrays guardados
mtx = data['mtx']
dist = data['dist']
rvecs = data['rvecs']
tvecs = data['tvecs']
# ============== Geometría del cuerpo (mm). Roja = primer punto ==============
objp = np.array([
    [0,  0, 0],   # ROJA
    [20, 0, 0],
    [20, 20, 0],
    [0,  20, 0]
], dtype=np.float32)

# ============== Parámetros de detección (tuneables) ==============
V_MIN     = 250     # piso de brillo; el código ajusta adaptativamente hacia arriba/abajo
AREA_MIN  = 25
AREA_MAX  = 12000
CIRC_MIN  = 0.55

S_MIN_RED = 30
H1_LOW, H1_HIGH = 0, 22
H2_LOW, H2_HIGH = 158, 180
RED_SCORE_MIN   = 10.0

# ============== Utils ==============
def circularity(cnt):
    a = cv2.contourArea(cnt); p = cv2.arcLength(cnt, True)
    if p <= 1e-6: return 0.0
    return float(4*np.pi*a/(p*p))

def ring_stats(img, center, radius, scale_in=0.3, scale_out=0.9):
    """Medianas en anillo (evita el highlight blanco del centro)."""
    H, W = img.shape[:2]
    cx, cy = int(center[0]), int(center[1])
    r_in  = max(1, int(radius*scale_in))
    r_out = max(r_in+1, int(radius*scale_out))
    y0, y1 = max(0, cy-r_out), min(H, cy+r_out+1)
    x0, x1 = max(0, cx-r_out), min(W, cx+r_out+1)
    if y1 <= y0 or x1 <= x0: return None
    crop = img[y0:y1, x0:x1]
    Y, X = np.ogrid[:crop.shape[0], :crop.shape[1]]
    m = (X-(cx-x0))**2 + (Y-(cy-y0))**2
    mask = (m <= r_out*r_out) & (m >= r_in*r_in)
    if not np.any(mask):
        # Fallback: parche 5x5
        x0s, x1s = max(cx-2,0), min(cx+3,W)
        y0s, y1s = max(cy-2,0), min(cy+3,H)
        crop2 = img[y0s:y1s, x0s:x1s]
        if crop2.size == 0: return None
        if crop2.ndim == 3: return np.median(crop2.reshape(-1,3), axis=0)
        return np.median(crop2.reshape(-1))
    if crop.ndim == 3:
        return np.median(crop[mask].reshape(-1,3), axis=0)
    else:
        return np.median(crop[mask].reshape(-1))

def red_score_at(frame_bgr, frame_hsv, center, approx_radius):
    bgr = ring_stats(frame_bgr, center, approx_radius)
    hsv = ring_stats(frame_hsv, center, approx_radius)
    if bgr is None or hsv is None: return -1e9
    b_med, g_med, r_med = bgr
    h_med, s_med, v_med = hsv
    hsv_is_red = ((H1_LOW <= h_med <= H1_HIGH) or (H2_LOW <= h_med <= H2_HIGH)) and (s_med >= S_MIN_RED)
    rgb_is_red = (r_med > g_med + 20) and (r_med > b_med + 20)
    score = (r_med - 0.5*(g_med + b_med)) + 0.05*s_med + (30 if hsv_is_red else 0) + (15 if rgb_is_red else 0)
    return float(score)

def order_ccw_from(pts, start_idx):
    P = np.array(pts, dtype=np.float32)
    C = P.mean(axis=0)
    ang = np.arctan2(P[:,1]-C[1], P[:,0]-C[0]); ang = np.where(ang<0, ang+2*np.pi, ang)
    a0 = ang[start_idx]; offs = (ang - a0) % (2*np.pi)
    return np.argsort(offs)

def solve_pnp_stable(objp, img_pts):
    flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", None)
    if flag is not None:
        ok, rvec, tvec = cv2.solvePnP(objp, img_pts, mtx, dist, flags=flag)
        if ok: return ok, rvec, tvec
    return cv2.solvePnP(objp, img_pts, mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)

def draw_point_safe(img, x, y, radius, color, label=None):
    try:
        xf = float(np.asarray(x).reshape(()))
        yf = float(np.asarray(y).reshape(()))
        if not (np.isfinite(xf) and np.isfinite(yf)): return False
        xi = int(np.rint(xf).item()); yi = int(np.rint(yf).item())
        h, w = img.shape[:2]
        xi = int(np.clip(xi, 0, w-1)); yi = int(np.clip(yi, 0, h-1))
        cv2.circle(img, (xi, yi), int(radius), tuple(int(c) for c in color), -1)
        if label:
            cv2.putText(img, label, (xi+10, yi-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return True
    except Exception as e:
        print("draw_point_safe: punto inválido:", type(x), type(y), e)
        return False

# ============== Detección principal ==============
def detectar_leds_y_roja(frame):
    """
    Devuelve:
      centers (lista de (x,y)),
      idx_red (o None),
      radios (lista),
      scores (lista),
      bright (máscara uint8 para debug)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    V = hsv[:,:,2].astype(np.uint8)

    # Umbral adaptativo en V con fallback
    p99 = float(np.percentile(V, 99))  # cuán brillante viene el frame
    th1 = int(max(min(p99 - 5, 255), V_MIN))   # primera pasada
    _, bright = cv2.threshold(V, th1, 255, cv2.THRESH_BINARY)
    if cv2.countNonZero(bright) < 10:
        p98 = float(np.percentile(V, 98))
        th2 = int(max(min(p98 - 5, 255), V_MIN - 20))  # relajamos un poco
        _, bright = cv2.threshold(V, th2, 255, cv2.THRESH_BINARY)

    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)

    cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cand = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < AREA_MIN or a > AREA_MAX:
            continue
        if circularity(c) < CIRC_MIN:
            continue
        M = cv2.moments(c);
        if M["m00"] == 0:
            continue
        cx, cy = M["m10"]/M["m00"], M["m01"]/M["m00"]
        r = math.sqrt(max(a,1)/np.pi)
        score = red_score_at(frame, hsv, (cx,cy), r)
        cand.append((a, (cx,cy), r, score))

    # Orden por área y recorte a top-4 (si hay menos, devuelvo las que haya)
    cand.sort(key=lambda x: x[0], reverse=True)
    cand = cand[:4] if len(cand) > 4 else cand

    centers = [c[1] for c in cand]
    radios  = [c[2] for c in cand]
    scores  = [c[3] for c in cand]

    idx_red = None
    if scores:
        i_max = int(np.argmax(scores))
        if scores[i_max] >= RED_SCORE_MIN:
            idx_red = i_max

    return centers, idx_red, radios, scores, bright

# ============== Main ==============
cap = cv2.VideoCapture(URL)
if not cap.isOpened():
    print("No se pudo abrir la cámara DroidCam"); raise SystemExit

print("Controles: g=guardar punzón | s=exportar CSV | m=mask | q=salir")
puntos_punzon = []
show_mask = False

while True:
    ok, frame = cap.read()
    if not ok:
        print("No se recibió frame"); break

    centers, idx_red, radios, scores, bright = detectar_leds_y_roja(frame)
    punzon_cam = None

    # Dibujar SIEMPRE lo que haya como candidatos
    for i, (pt, r) in enumerate(zip(centers, radios)):
        color = (0,0,255) if (idx_red is not None and i == idx_red) else (0,255,255)
        draw_point_safe(frame, pt[0], pt[1], 8, color)

    # Texto de ayuda
    cv2.putText(frame, f"Cands: {len(centers)}  Roja: {idx_red if idx_red is not None else '-'}",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Si hay 4 y roja -> PnP + punzón
    if len(centers) == 4 and idx_red is not None:
        order = order_ccw_from(centers, idx_red)
        centers_ord = np.array([centers[i] for i in order], dtype=np.float32)

        ok_pnp, rvec, tvec = solve_pnp_stable(objp, centers_ord)
        if ok_pnp:
            proj, _ = cv2.projectPoints(objp, rvec, tvec, mtx, dist)
            proj = np.asarray(proj, dtype=np.float64).reshape(-1,2)
            for i,(x,y) in enumerate(proj):
                color = (0,0,255) if i==0 else (0,255,255)
                cv2.circle(frame, (int(round(x)), int(round(y))), 7, color, 2)

            # Punzón: 10 cm desde la roja en dir (centroide->roja)
            centroide_obj = objp.mean(axis=0, keepdims=True)
            red_obj = objp[0:1]
            v = (red_obj - centroide_obj).astype(np.float32)
            n = np.linalg.norm(v)
            if n > 1e-6:
                u = v / n
                punzon_obj = (red_obj + u * 100.0).astype(np.float32).reshape(1,3)
                p2d, _ = cv2.projectPoints(punzon_obj, rvec, tvec, mtx, dist)
                p2d = np.asarray(p2d, dtype=np.float64).reshape(-1,2)
                if draw_point_safe(frame, p2d[0,0], p2d[0,1], 10, (255,0,255), "Punzón"):
                    R, _ = cv2.Rodrigues(rvec)
                    punzon_cam = (R @ punzon_obj.T + tvec).T[0]

    # Mostrar
    cv2.imshow("DroidCam PnP (lite v2)", frame)
    if show_mask:
        cv2.imshow("mask_bright(V)", bright)
    else:
        if cv2.getWindowProperty("mask_bright(V)", 0) != -1:
            cv2.destroyWindow("mask_bright(V)")

    k = cv2.waitKey(1) & 0xFF
    if k == ord('g'):
        if punzon_cam is not None and np.all(np.isfinite(punzon_cam)):
            puntos_punzon.append(punzon_cam.tolist())
            print("Punzón guardado:", np.round(punzon_cam, 2))
        else:
            print("No se pudo guardar: faltan 4 LEDs + roja/PnP.")
    elif k == ord('s'):
        if puntos_punzon:
            np.savetxt("punzones_3d_en_camara.csv", np.array(puntos_punzon),
                       delimiter=",", header="X,Y,Z", comments='', fmt="%.3f")
            print(f"CSV guardado ({len(puntos_punzon)} filas).")
        else:
            print("No hay punzones para guardar.")
    elif k == ord('m'):
        show_mask = not show_mask
    elif k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()