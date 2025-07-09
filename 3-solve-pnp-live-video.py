import cv2
import numpy as np

# este es el codigo general que deberia correrse para ver el video de la deteccion del beacon en vivo.
# si quiero marcar un punto de la anatomia del paciente, como para la calibracion por 3 puntos, tambien los guarda
# para su procesamiento posterior en 5-registro-3D.py

# --- Parámetros de la cámara y objeto ---
objp = np.array([[0, 0, 0],
                 [1, 0, 0],
                 [1, 1, 0],
                 [0, 1, 0]], dtype=np.float32)

mtx = np.array([[2992.45904, 0, 1489.42745],
                [0, 2979.52349, 2003.95063],
                [0, 0, 1]], dtype=np.float32)

dist = np.array([0.2433, -1.2952, -0.0025, -0.0020, 2.41], dtype=np.float32)

# --- Almacenamiento de puntos seleccionados ---
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

    led_centers = detectar_leds_automaticamente(frame)
    if led_centers is not None:
        ret, rvecs, tvecs = cv2.solvePnP(objp, led_centers, mtx, dist)
        if ret:
            # Dibujar puntos proyectados
            projected_points, _ = cv2.projectPoints(objp, rvecs, tvecs, mtx, dist)
            for pt in projected_points:
                x, y = pt.ravel()
                cv2.circle(frame, (int(x), int(y)), 6, (0, 255, 0), -1)

            # Calcular el centroide 3D y mostrarlo
            centroide_3d = np.mean(objp, axis=0).reshape(1, 3)
            centroide_camara = (cv2.Rodrigues(rvecs)[0] @ centroide_3d.T + tvecs).T[0]

            # Dibujar centro proyectado
            centroide_2d, _ = cv2.projectPoints(centroide_3d, rvecs, tvecs, mtx, dist)
            cx, cy = centroide_2d.ravel()
            if np.all(np.isfinite([cx, cy])):
                cv2.circle(frame, (int(cx), int(cy)), 10, (0, 0, 255), -1)
                cv2.putText(frame, "Centro", (int(cx)+10, int(cy)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Mostrar video
    cv2.imshow("PnP en vivo", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('g'):
        if led_centers is not None and ret:
            puntos_guardados.append(centroide_camara.tolist())
            print(f"Punto guardado: {centroide_camara}")
        else:
            print("No se pudo guardar el punto: detección inválida")

    elif key == ord('s'):
        if puntos_guardados:
            np.savetxt("puntos_3d_en_camara.csv", np.array(puntos_guardados),
                       delimiter=",", header="X,Y,Z", comments='')
            print(f"Archivo guardado con {len(puntos_guardados)} puntos.")
        else:
            print("No hay puntos para guardar.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()