import cv2
import numpy as np

# --- Coordenadas 3D conocidas ---
objp = np.array([
    [0, 0, 0],        # LED ROJO
    [150, 0, 0],      # LED BLANCO
    [150, 150, 0],    # LED BLANCO
    [0, 150, 0]       # LED BLANCO
], dtype=np.float32)

# --- Calibración de cámara ---
# Cargar el archivo .npz
data = np.load('B.npz')
# Acceder a los arrays guardados
mtx = data['mtx']
dist = data['dist']
rvecs = data['rvecs']
tvecs = data['tvecs']

# --- Variables globales para tracking ---
puntos_guardados = []
tracking_leds = {
    "rojo": None,
    "blanco1": None,
    "blanco2": None,
    "blanco3": None
}
MAX_DIST_PIXEL = 50

# --- Función para tracking ---
def actualizar_tracker(nuevos_puntos, colores=["rojo", "blanco1", "blanco2", "blanco3"]):
    global tracking_leds
    nuevos_puntos = dict(zip(colores, nuevos_puntos))
    for color, nuevo_pt in nuevos_puntos.items():
        prev_pt = tracking_leds[color]
        if prev_pt is None:
            tracking_leds[color] = nuevo_pt
        else:
            dist = np.linalg.norm(np.array(prev_pt) - np.array(nuevo_pt))
            if dist < MAX_DIST_PIXEL:
                tracking_leds[color] = 0.6 * np.array(prev_pt) + 0.4 * np.array(nuevo_pt)
    return np.array([tracking_leds[c] for c in colores], dtype=np.float32)

 # --- ROI para optimizar búsqueda ---
def calcular_roi(centros, margen=50):
     if centros is None:
         return None
     centros = np.array(centros)
     x_min = max(int(np.min(centros[:, 0]) - margen), 0)
     x_max = int(np.max(centros[:, 0]) + margen)
     y_min = max(int(np.min(centros[:, 1]) - margen), 0)
     y_max = int(np.max(centros[:, 1]) + margen)
     return (x_min, x_max, y_min, y_max)

# --- Detectar LED rojo por color ---
def detectar_rojo(imagen, roi=None):
    if roi:
        x_min, x_max, y_min, y_max = roi
        imagen = imagen[y_min:y_max, x_min:x_max]

    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    bajo = np.array([0, 100, 100])
    alto = np.array([10, 255, 255])
    mascara = cv2.inRange(hsv, bajo, alto)

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contornos:
        cnt = max(contornos, key=cv2.contourArea)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            if roi:
                cx += x_min
                cy += y_min
            return (cx, cy)
    return None

# --- Detectar LEDs blancos por brillo ---
# --- Detectar LEDs blancos por contorno ---
def detectar_blancos(imagen, cantidad=3, roi=None):
    if roi:
        x_min, x_max, y_min, y_max = roi
        imagen_roi = imagen[y_min:y_max, x_min:x_max]
    else:
        imagen_roi = imagen

    # Convertir a gris y suavizar
    gray = cv2.cvtColor(imagen_roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Umbral alto para detectar zonas muy brillantes (LEDs blancos)
    _, thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY)

    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ordenar por área de mayor a menor y tomar solo los más grandes
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:cantidad]

    puntos = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            if roi:
                cx += x_min
                cy += y_min
            puntos.append((cx, cy))

    # Ordenar en el mismo criterio que antes (primero por Y, luego por X)
    puntos = sorted(puntos, key=lambda p: (p[1], p[0]))

    return puntos if len(puntos) == cantidad else None



# --- Captura ---
# Reemplaza con la IP y puerto que muestra la app DroidCam en tu iPhone
ip = "192.168.0.91"  # Ejemplo: 192.168.1.139
port = "4747"       # Puerto por defecto que usa DroidCam

# URL para video en formato MJPEG proporcionado por DroidCam
url = f"http://{ip}:{port}/video"
cap = cv2.VideoCapture(url)
if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

roi = None
print("Presioná 'g' para guardar el punto actual")
print("Presioná 's' para guardar archivo CSV final")
print("Presioná 'q' para salir")

while True:
    success, frame = cap.read()
    if not success:
        break

    rojo = detectar_rojo(frame, roi)
    blancos = detectar_blancos(frame, cantidad=3, roi=roi)

    if rojo and blancos:
        leds = [rojo] + blancos  # orden: rojo, blanco1, blanco2, blanco3

        leds = actualizar_tracker(leds)
        roi = calcular_roi(leds, margen=60)
        success_pnp, rvecs, tvecs, inliers = cv2.solvePnPRansac(
            objp, leds, mtx, dist,
            reprojectionError=8.0, confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if success_pnp and inliers is not None and len(inliers) >= 4:
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
            if success_pnp:
                colores_bgr = [(0, 0, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255)]
                for (x, y), color in zip(leds, colores_bgr):
                    cv2.circle(frame, (int(x), int(y)), 6, color, -1)

    cv2.imshow("PnP en vivo", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('g'):
        if leds is not None and success_pnp:
            puntos_guardados.append(tvecs.ravel().tolist())
            print(f"Punto guardado: {tvecs.ravel()}")
    elif key == ord('s'):
        if puntos_guardados:
            np.savetxt("puntos_3d_en_camara.csv", np.array(puntos_guardados),
                       delimiter=",", header="X,Y,Z", comments='', fmt="%.3f")
            print(f"Archivo guardado con {len(puntos_guardados)} puntos.")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()












