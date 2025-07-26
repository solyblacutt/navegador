import cv2
import numpy as np

# ---------- Parámetros cámara / objeto ----------
OBJ_PTS = np.array([[0, 0, 0],
                    [20, 0, 0],
                    [20, 20, 0],
                    [0, 20, 0]], dtype=np.float32)

CAM_MTX = np.array([[2992.45904, 0, 1489.42745],
                    [0, 2979.52349, 2003.95063],
                    [0, 0, 1]], dtype=np.float32)

CAM_DIST = np.array([0.2433, -1.2952, -0.0025, -0.0020, 2.41], dtype=np.float32)

# ---------- Rangos HSV para cada color ----------
RANGO_VERDE   = ([45,  80, 80], [ 85, 255, 255])
RANGO_AMARILLO= ([20,  80, 80], [ 35, 255, 255])
RANGOS_ROJO   = [([0,  80, 80], [10, 255, 255]),
                 ([160,80, 80], [180,255,255])]

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

def detectar_leds_color(frame: np.ndarray):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Verde
    m_verde = cv2.inRange(hsv, np.array(RANGO_VERDE[0]), np.array(RANGO_VERDE[1]))
    verdes  = _mask2centers(m_verde, 1)

    # Amarillo
    m_amar  = cv2.inRange(hsv, np.array(RANGO_AMARILLO[0]), np.array(RANGO_AMARILLO[1]))
    amarillos = _mask2centers(m_amar, 1)

    # Rojo (dos rangos)
    m_rojo1 = cv2.inRange(hsv, np.array(RANGOS_ROJO[0][0]), np.array(RANGOS_ROJO[0][1]))
    m_rojo2 = cv2.inRange(hsv, np.array(RANGOS_ROJO[1][0]), np.array(RANGOS_ROJO[1][1]))
    rojos   = _mask2centers(cv2.bitwise_or(m_rojo1, m_rojo2), 2)

    if not (len(verdes) == len(amarillos) == 1 and len(rojos) == 2):
        return None            # detección incompleta

    # Ordenar los dos rojos de izquierda a derecha
    rojos.sort(key=lambda p: p[0])

    leds = verdes + amarillos + rojos      # verde, amarillo, rojo1, rojo2
    return np.array(leds, dtype=np.float32)

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

    led_centers = detectar_leds_color(frame)
    if led_centers is not None:
        ok, rvec, tvec = cv2.solvePnP(OBJ_PTS, led_centers, CAM_MTX, CAM_DIST)
        if ok:
            # dibujar vértices proyectados
            proj, _ = cv2.projectPoints(OBJ_PTS, rvec, tvec, CAM_MTX, CAM_DIST)
            for (x, y) in proj.reshape(-1, 2):
                cv2.circle(frame, (int(x), int(y)), 6, (0, 255, 0), -1)

            # centroide proyectado
            centro3d = OBJ_PTS.mean(axis=0).reshape(1, 3)
            centro2d, _ = cv2.projectPoints(centro3d, rvec, tvec, CAM_MTX, CAM_DIST)
            cx, cy = map(int, centro2d.ravel())
            cv2.circle(frame, (cx, cy), 10, (0, 0, 0), -1)
            cv2.putText(frame, "Centro", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 2)

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
