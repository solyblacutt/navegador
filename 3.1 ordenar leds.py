import cv2
import numpy as np

# ---------- Parámetros cámara / objeto ----------
OBJ_PTS = np.array([[0, 0, 0],
                    [20, 0, 0],
                    [20, 20, 0],
                    [0, 20, 0]], dtype=np.float32)

CAM_MTX = np.array([[1.23994321e+03, 0.00000000e+00, 9.42066048e+02],
 [0.00000000e+00, 1.24162269e+03, 5.16545687e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)

CAM_DIST = np.array([ 0.02489004,  0.12455246, -0.01055148,  0.00068239,  0.14485304], dtype=np.float32)


# ---------- Detección de blobs por color ----------
def _mask2centers(mask: np.ndarray, max_cnts=2):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:max_cnts]
    centers = []
    for c in cnts:
        m = cv2.moments(c)
        if m["m00"]:
            centers.append((int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])))
    return centers

# ---------- rangos HSV de colores ----------
R_H_LOW  = np.array([  0,  80, 120])   # rojo: zona baja   H 0-10
R_H_HIGH = np.array([ 10, 255,255])
R2_H_LOW = np.array([170, 80, 120])    # rojo: zona alta  H 170-180
R2_H_HIGH= np.array([180,255,255])

Y_H_LOW  = np.array([ 20, 80, 120])    # amarillo
Y_H_HIGH = np.array([ 35,255,255])

G_H_LOW  = np.array([ 55, 40, 65])    # verde
G_H_HIGH = np.array([ 85,255,255])

def _clasifica_color(hsv_val):
    h,s,v = hsv_val
    if ((R_H_LOW<=hsv_val).all()  and (hsv_val<=R_H_HIGH).all()) \
    or ((R2_H_LOW<=hsv_val).all() and (hsv_val<=R2_H_HIGH).all()):
        return "rojo"
    if (Y_H_LOW<=hsv_val).all() and (hsv_val<=Y_H_HIGH).all():
        return "amarillo"
    if (G_H_LOW<=hsv_val).all() and (hsv_val<=G_H_HIGH).all():
        return "verde"
    return "otro"

# ---------- detección por contorno + clasificación HSV ----------
def detectar_leds_contorno_color(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Otsu para binarizar las zonas brillantes
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # limpiar ruido
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)

    cnts,_ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts   = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]

    if len(cnts) != 4:
        return None          # detección incompleta

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    leds_dict = {"verde":[], "amarillo":[], "rojo":[]}

    for c in cnts:
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx = int(m["m10"]/m["m00"])
        cy = int(m["m01"]/m["m00"])
        color = _clasifica_color(hsv[cy, cx])
        if color in leds_dict:
            leds_dict[color].append((cx, cy))

    # necesitamos 1 verde, 1 amarillo y 2 rojos
    if not (len(leds_dict["verde"])==1 and len(leds_dict["amarillo"])==1 and len(leds_dict["rojo"])==2):
        return None

    # ordenar los 2 rojos de izquierda a derecha
    leds_dict["rojo"].sort(key=lambda p: p[0])

    # construir orden final:  verde → amarillo → rojo izq → rojo der
    orden = leds_dict["verde"] + leds_dict["amarillo"] + leds_dict["rojo"]
    return np.array(orden, dtype=np.float32)

# ---------- Captura en vivo ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("No se pudo abrir la cámara")

puntos_guardados = []
print("[g] guardar centroide   [s] guardar CSV   [q] salir")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    led_centers = detectar_leds_contorno_color(frame)
    print(led_centers)
    if led_centers is not None:
        ok, rvec, tvec = cv2.solvePnP(OBJ_PTS, led_centers, CAM_MTX, CAM_DIST)
        if ok:
            # dibujar vértices proyectados
            proj, _ = cv2.projectPoints(OBJ_PTS, rvec, tvec, CAM_MTX, CAM_DIST)
            for (x, y) in proj.reshape(-1, 2):
                cv2.circle(frame, (int(x), int(y)), 6, (0, 255, 0), -1)

            # ------------- centroide proyectado -------------
            centro3d = OBJ_PTS.mean(axis=0).reshape(1, 3)
            centro2d, _ = cv2.projectPoints(centro3d, rvec, tvec, CAM_MTX, CAM_DIST)
            cx_raw, cy_raw = centro2d.ravel()  # np.float32

            # Si salen valores no finitos, salteo el frame
            if not np.isfinite([cx_raw, cy_raw]).all():
                print("⚠️ Centroide no finito, se omite frame")
                continue  # vuelve al while

            cx = int(round(float(cx_raw)))
            cy = int(round(float(cy_raw)))

            h, w = frame.shape[:2]
            if 0 <= cx < w and 0 <= cy < h:
                cv2.circle(frame, (cx, cy), 10, (0, 0, 0), -1)
                cv2.putText(frame, "Centro", (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            else:
                print("⚠️ Centroide fuera de la imagen:", cx, cy)


    # ---------- interfaz ----------
    cv2.imshow("Tracking 4-LEDs (color)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('g') and led_centers is not None and ok:
        cam_centroid = (cv2.Rodrigues(rvec)[0] @ centro3d.T + tvec).T[0]
        puntos_guardados.append(cam_centroid)
        print("Punto guardado:", cam_centroid.round(3))
    elif key == ord('s'):
        if puntos_guardados:
            np.savetxt("puntos_3d_en_camara.csv", np.array(puntos_guardados),
                       delimiter=",", header="X,Y,Z", fmt="%.3f", comments='')
            print("CSV guardado.")
    elif key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()




"""

import cv2
import numpy as np

# ---------- rangos HSV de colores ----------
R_H_LOW  = np.array([  0,  80, 120])   # rojo: zona baja   H 0-10
R_H_HIGH = np.array([ 10, 255,255])
R2_H_LOW = np.array([170, 80, 120])    # rojo: zona alta  H 170-180
R2_H_HIGH= np.array([180,255,255])

Y_H_LOW  = np.array([ 20, 80, 120])    # amarillo
Y_H_HIGH = np.array([ 35,255,255])

G_H_LOW  = np.array([ 55, 40,  80])    # verde
G_H_HIGH = np.array([ 85,255,255])

def _clasifica_color(hsv_val):
    h,s,v = hsv_val
    if ((R_H_LOW<=hsv_val).all()  and (hsv_val<=R_H_HIGH).all()) \
    or ((R2_H_LOW<=hsv_val).all() and (hsv_val<=R2_H_HIGH).all()):
        return "rojo"
    if (Y_H_LOW<=hsv_val).all() and (hsv_val<=Y_H_HIGH).all():
        return "amarillo"
    if (G_H_LOW<=hsv_val).all() and (hsv_val<=G_H_HIGH).all():
        return "verde"
    return "otro"

# ---------- detección por contorno + clasificación HSV ----------
def detectar_leds_contorno_color(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Otsu para binarizar las zonas brillantes
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # limpiar ruido
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)

    cnts,_ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts   = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]

    if len(cnts) != 4:
        return None          # detección incompleta

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    leds_dict = {"verde":[], "amarillo":[], "rojo":[]}

    for c in cnts:
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx = int(m["m10"]/m["m00"])
        cy = int(m["m01"]/m["m00"])
        color = _clasifica_color(hsv[cy, cx])
        if color in leds_dict:
            leds_dict[color].append((cx, cy))

    # necesitamos 1 verde, 1 amarillo y 2 rojos
    if not (len(leds_dict["verde"])==1 and len(leds_dict["amarillo"])==1 and len(leds_dict["rojo"])==2):
        return None

    # ordenar los 2 rojos de izquierda a derecha
    leds_dict["rojo"].sort(key=lambda p: p[0])

    # construir orden final:  verde → amarillo → rojo izq → rojo der
    orden = leds_dict["verde"] + leds_dict["amarillo"] + leds_dict["rojo"]
    return np.array(orden, dtype=np.float32)
    
    """