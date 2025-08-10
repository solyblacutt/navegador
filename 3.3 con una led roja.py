
"""
import cv2
import numpy as np

# --- Parámetros de la cámara y objeto ---
objp = np.array([[0, 0, 0],
                 [20, 0, 0],
                 [20, 20, 0],
                 [0, 20, 0]], dtype=np.float32)

mtx = np.array([[1.23994321e+03, 0.00000000e+00, 9.42066048e+02],
 [0.00000000e+00, 1.24162269e+03, 5.16545687e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)

dist = np.array([0.02489004, 0.12455246, -0.01055148, 0.00068239, 0.14485304], dtype=np.float32)

puntos_guardados = []

# --- Función para detectar LEDs y su color ---
def detectar_leds_automaticamente(imagen):
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]

    leds = []
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Color promedio en HSV alrededor del LED
            radio = 3
            patch = hsv[max(0, cy-radio):cy+radio+1, max(0, cx-radio):cx+radio+1]
            mean_color = np.mean(patch.reshape(-1, 3), axis=0)  # H, S, V

            # Rango para detectar rojo en HSV
            rojo_bajo1 = np.array([0, 100, 100])
            rojo_alto1 = np.array([10, 255, 255])
            rojo_bajo2 = np.array([160, 100, 100])
            rojo_alto2 = np.array([179, 255, 255])

            es_rojo = (np.any(cv2.inRange(patch, rojo_bajo1, rojo_alto1)) or
                       np.any(cv2.inRange(patch, rojo_bajo2, rojo_alto2)))

            leds.append({"pos": (cx, cy), "rojo": es_rojo})

    if len(leds) != 4:
        return None

    # Ordenar de izquierda a derecha
    leds = sorted(leds, key=lambda x: x["pos"][0])

    # Asegurar que el LED rojo esté identificado
    posiciones = np.array([led["pos"] for led in leds], dtype=np.float32)
    return posiciones, leds

# --- Captura de video ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

print("Presioná 'g' para guardar el punto actual")
print("Presioná 's' para guardar archivo CSV final")
print("Presioná 'q' para salir")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    deteccion = detectar_leds_automaticamente(frame)
    if deteccion is not None:
        led_centers, leds_info = deteccion

        ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(
            objp, led_centers, mtx, dist,
            reprojectionError=8.0, confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if ret and inliers is not None and len(inliers) >= 4:
            projected_points, _ = cv2.projectPoints(objp, rvecs, tvecs, mtx, dist)
            for i, pt in enumerate(projected_points):
                x, y = pt.ravel()
                color = (0, 0, 255) if leds_info[i]["rojo"] else (0, 255, 0)
                cv2.circle(frame, (int(x), int(y)), 6, color, -1)

            # Dibujar puntos proyectados
            projected_points, _ = cv2.projectPoints(objp, rvecs, tvecs, mtx, dist)
            for pt in projected_points:
                x, y = pt.ravel()
                cv2.circle(frame, (int(x), int(y)), 6, (0, 255, 0), -1)
            # Calcular el centroide 3D y mostrarlo
            centroide_3d = np.mean(objp, axis=0).reshape(1, 3)
            centroide_camara = (cv2.Rodrigues(rvecs)[0] @ centroide_3d.T + tvecs).T[0]
            punzon_obj = np.array([[-10, 10, 0]], dtype=np.float32)
            punzon_camara = (cv2.Rodrigues(rvecs)[0] @ punzon_obj.T + tvecs).T[0]
            # Dibujar centro proyectado
            centroide_2d, _ = cv2.projectPoints(centroide_3d, rvecs, tvecs, mtx, dist)
            cx, cy = centroide_2d.ravel()
            if np.all(np.isfinite([cx, cy])):
                cv2.circle(frame, (int(cx), int(cy)), 10, (0, 0, 255), -1)
                cv2.putText(frame, "Centro", (int(cx) + 10, int(cy) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            punzon_2d, _ = cv2.projectPoints(punzon_obj, rvecs, tvecs, mtx, dist)
            px, py = punzon_2d.ravel()
            if np.all(np.isfinite([px, py])):
                cv2.circle(frame, (int(px), int(py)), 8, (255, 0, 255), -1)
                cv2.putText(frame, f"{int(px)}, {int(py)}", (int(px) + 10, int(py) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    # Mostrar video
    cv2.imshow("PnP en vivo", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('g'):
        if deteccion is not None and ret:
            puntos_guardados.append(tvecs.ravel().tolist())
            print(f"Punto guardado: {tvecs.ravel()}")
        else:
            print("No se pudo guardar el punto: detección inválida")

    elif key == ord('s'):
        if puntos_guardados:
            np.savetxt("puntos_3d_en_camara.csv", np.array(puntos_guardados),
                       delimiter=",", header="X,Y,Z", comments='', fmt="%.3f")
            print(f"Archivo guardado con {len(puntos_guardados)} puntos.")
        else:
            print("No hay puntos para guardar.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

"""





import cv2
import numpy as np
from itertools import permutations

# --- TU MODELO (mm) ---
objp = np.array([[0, 0, 0],
                 [20, 0, 0],
                 [20, 20, 0],
                 [0, 20, 0]], dtype=np.float32)

# Cargar el archivo .npz
data = np.load('B.npz')
# Acceder a los arrays guardados
mtx = data['mtx']
dist = data['dist']
rvecs = data['rvecs']
tvecs = data['tvecs']

IP   = "192.168.0.91"
PORT = "4747"
URL  = f"http://{IP}:{PORT}/video"   # stream MJPEG de DroidCam

# --- 1) Detección por contorno (brillo) ---
def detectar_leds_por_contorno(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # más robusto que un umbral fijo: Otsu
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # limpia ruido muy pequeño
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)

    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]

    centers = []
    for c in cnts:
        if cv2.contourArea(c) < 10:  # evita chispas
            continue
        m = cv2.moments(c)
        if m["m00"]:
            cx = int(m["m10"]/m["m00"])
            cy = int(m["m01"]/m["m00"])
            centers.append((cx, cy))

    if len(centers) != 4:
        return None
    return np.array(centers, dtype=np.float32)

# Rangos HSV para rojo (OpenCV: H 0–179)
ROJO1_LOW  = np.array([  50, 40,  0], dtype=np.uint8)
ROJO1_HIGH = np.array([ 90,255, 255], dtype=np.uint8)


def indice_led_roja(frame, centers, k=7):
    """
    Devuelve el índice (0..3) del centro que es rojo, o None si no encuentra.
    Toma un patch kxk alrededor del centro para robustez, promedia en HSV y
    evalúa contra los rangos de rojo.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, W = hsv.shape[:2]

    for i, (cx, cy) in enumerate(centers.astype(int)):
        # recorte del parche alrededor del centro
        x0, x1 = max(0, cx - k), min(W, cx + k + 1)
        y0, y1 = max(0, cy - k), min(H, cy + k + 1)
        patch = hsv[y0:y1, x0:x1]

        if patch.size == 0:
            continue

        hsv_mean = patch.reshape(-1, 3).mean(axis=0)           # float64
        pixel = hsv_mean.astype(np.uint8).reshape(1, 1, 3)     # (1,1,3) uint8

        in_r1 = cv2.inRange(pixel, ROJO1_LOW, ROJO1_HIGH)[0, 0] > 0

        #Debug opcional:
        print("HSV medio idx", i, "=", hsv_mean, "r1:", in_r1)

        if in_r1:
            return i

    return None

# --- 3) PnP con permutaciones, forzando que el 2D "rojo" mapee a ALGUNA esquina ---
def solvepnp_con_roja(centers2d, red_idx, reproj_tol_px=5.0):
    best = None
    best_err = 1e9
    idxs = [0,1,2,3]
    for perm in permutations(idxs):
        # perm dice cómo ordenar centers2d para que correspondan a objp
        # imponemos que el punto 'red_idx' (en image) vaya a algún vértice (cualquiera)
        # (si tu LED rojo es SIEMPRE una esquina fija, podés restringirla aquí)
        ordered = centers2d[list(perm)]
        ok, rvec, tvec = cv2.solvePnP(objp, ordered, mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        reproj, _ = cv2.projectPoints(objp, rvec, tvec, mtx, dist)
        err = np.mean(np.linalg.norm(reproj.reshape(-1,2) - ordered, axis=1))
        if err < best_err:
            best_err = err
            best = (rvec, tvec, perm)
    if best is None or best_err > reproj_tol_px:
        return None, None, None, best_err
    return best[0], best[1], best[2], best_err

# --- 4) Construir punzón a 10 cm desde la LED roja (en marco del OBJETO) ---
def punzon_desde_roja(red_obj, distancia_mm=100.0, modo="radial"):
    """
    red_obj: (3,) punto 3D de la LED roja en el objeto
    modo:
      - 'radial': desde el centro del cuadrado hacia la roja, en el plano
      - 'x'     : +X del objeto
      - 'y'     : +Y del objeto
      - 'normal': +Z del objeto (sale del plano)
    """
    centro_obj = objp.mean(axis=0)
    if modo == "radial":
        v = red_obj - centro_obj
        v[2] = 0.0
        n = np.linalg.norm(v[:2])
        if n < 1e-6:
            v[:2] = np.array([1.0, 0.0])  # fallback
            n = 1.0
        dir2d = v[:2] / n
        return red_obj + np.array([dir2d[0]*distancia_mm, dir2d[1]*distancia_mm, 0.0], dtype=np.float32)
    elif modo == "x":
        return red_obj + np.array([distancia_mm, 0.0, 0.0], dtype=np.float32)
    elif modo == "y":
        return red_obj + np.array([0.0, distancia_mm, 0.0], dtype=np.float32)
    elif modo == "normal":
        return red_obj + np.array([0.0, 0.0, distancia_mm], dtype=np.float32)
    else:
        return red_obj.copy()

# ===========================
#   LOOP DE VÍDEO (ejemplo)
# ===========================
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(URL)
if not cap.isOpened():
    raise SystemExit("No se pudo abrir la cámara")

print("q para salir")
while True:
    ok, frame = cap.read()
    if not ok:
        break

    centers = detectar_leds_por_contorno(frame)
    if centers is not None:
        ridx = indice_led_roja(frame, centers)
        if ridx is None:
            print("led roja es none")
        if ridx is not None:
            rvec, tvec, perm, err = solvepnp_con_roja(centers, ridx, reproj_tol_px=8)
            if rvec is not None:
                # dibujar las 4 proyecciones del modelo
                proj, _ = cv2.projectPoints(objp, rvec, tvec, mtx, dist)
                for (x, y) in proj.reshape(-1,2):
                    cv2.circle(frame, (int(x), int(y)), 6, (0,255,0), -1)

                # ¿qué vértice del objeto quedó como “rojo”?
                obj_red_idx = perm.index(ridx)  # dónde cayó el punto rojo en el orden de objp
                red_obj = objp[obj_red_idx].astype(np.float32)

                # punzón a 10 cm desde la LED roja (modo radial en el plano)
                punzon_obj = punzon_desde_roja(red_obj, distancia_mm=100.0, modo="radial")

                # proyectar centroide y punzón
                centro_obj = objp.mean(axis=0, dtype=np.float32).reshape(1,3)
                centro_2d, _ = cv2.projectPoints(centro_obj, rvec, tvec, mtx, dist)
                cx, cy = map(int, centro_2d.ravel())
                cv2.circle(frame, (cx, cy), 10, (0,0,255), -1)
                cv2.putText(frame, "Centro", (cx+10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,255), 2)

                punzon_2d, _ = cv2.projectPoints(punzon_obj.reshape(1,3), rvec, tvec, mtx, dist)
                px, py = map(int, punzon_2d.ravel())
                cv2.circle(frame, (px, py), 8, (255, 0, 255), -1)  # rosado
                cv2.putText(frame, "Punzón", (px+10, py-10), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,0,255), 2)

                # marca la LED roja detectada (2D)
                rx, ry = centers[ridx].astype(int)
                cv2.circle(frame, (rx, ry), 7, (0,0,255), 2)

    cv2.imshow("PnP en vivo (roja -> punzón 10cm)", frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
