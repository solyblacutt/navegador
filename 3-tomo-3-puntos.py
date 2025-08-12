import cv2
import numpy as np

# este es el codigo general que deberia correrse para marcar un punto de la anatomia del paciente y guardarlos
# para su procesamiento posterior en 5-registro-3D.py

# --- Parámetros de la cámara y objeto --- 20 mm
objp = np.array([[0, 0, 0],
                 [150, 0, 0],
                 [150, 150, 0],
                 [0, 150, 0]], dtype=np.float32)
"""
mtx = np.array([[1.23994321e+03, 0.00000000e+00, 9.42066048e+02],
 [0.00000000e+00, 1.24162269e+03, 5.16545687e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)

dist = np.array([ 0.02489004,  0.12455246, -0.01055148,  0.00068239,  0.14485304], dtype=np.float32)
"""

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


# --- Almacenamieto de puntos seleccionados ---
puntos_guardados = []

# --- Función para detectar LEDs ---
def detectar_leds_automaticamente(imagen):
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]

    leds = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            leds.append((cx, cy))

    leds = np.array(leds, dtype=np.float32)
    if len(leds) != 4:
        return None

    # Ordenar en sentido horario
    c = np.mean(leds, axis=0)
    angles = np.arctan2(leds[:, 1] - c[1], leds[:, 0] - c[0])
    leds = leds[np.argsort(angles)]
    return leds

# --- Captura de video ---
cap = cv2.VideoCapture(URL)
#cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara DroidCam")
    exit()

print("Presioná 'g' para guardar el punto actual")
print("Presioná 's' para guardar archivo CSV final")
print("Presioná 'q' para salir")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    led_centers = detectar_leds_automaticamente(frame)
    if led_centers is not None:
        #ret, rvecs, tvecs = cv2.solvePnP(objp, led_centers, mtx, dist)
        ret, rvecs, tvecs, inliers = cv2.solvePnPRansac( objp, led_centers, mtx, dist, reprojectionError=8.0, confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE)
        if ret and inliers is not None and len(inliers) >= 4:
            # Dibujar puntos proyectados
            projected_points, _ = cv2.projectPoints(objp, rvecs, tvecs, mtx, dist)
            for pt in projected_points:
                x, y = pt.ravel()
                cv2.circle(frame, (int(x), int(y)), 6, (0, 255, 0), -1)

            # Calcular el centroide 3D y mostrarlo
            centroide_3d = np.mean(objp, axis=0).reshape(1, 3)
            centroide_camara = (cv2.Rodrigues(rvecs)[0] @ centroide_3d.T + tvecs).T[0]
            punzon_obj = np.array([[-110, 110, 150]], dtype=np.float32)
            punzon_camara = (cv2.Rodrigues(rvecs)[0] @ punzon_obj.T + tvecs).T[0]

            # Dibujar centro proyectado
            centroide_2d, _ = cv2.projectPoints(centroide_3d, rvecs, tvecs, mtx, dist)
            cx, cy = centroide_2d.ravel()
            if np.all(np.isfinite([cx, cy])):
                cv2.circle(frame, (int(cx), int(cy)), 10, (0, 0, 255), -1)
                cv2.putText(frame, "Centro", (int(cx)+10, int(cy)-10),
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
        if led_centers is not None and ret:
            puntos_guardados.append(punzon_camara.tolist()) # ES PUNZON?CAMARA A GUARDARRRR
            print(f"Punto guardado: {punzon_camara}")
            print(rvecs)
            print(tvecs)
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