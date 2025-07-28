import cv2
import numpy as np

# este script es una copia modificada de 3-tomo-3-puntos.py

# --- Parámetros de la cámara y objeto --- 20 mm

# Asumimos un cuadrado de 20x20 mm, con este orden:
# Verde (0,0), Rojo (20,0), Rojo (20,20), Amarillo (0,20)
objp = np.array([
    [0, 0, 0],      # LED VERDE
    [20, 0, 0],     # LED ROJO
    [20, 20, 0],    # LED ROJO
    [0, 20, 0]      # LED AMARILLO
], dtype=np.float32)


mtx = np.array([[1.23994321e+03, 0.00000000e+00, 9.42066048e+02],
 [0.00000000e+00, 1.24162269e+03, 5.16545687e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)

dist = np.array([ 0.02489004,  0.12455246, -0.01055148,  0.00068239,  0.14485304], dtype=np.float32)

# --- Almacenamieto de puntos seleccionados ---
puntos_guardados = []

# --- Función para detectar LEDs ---
def detectar_leds_por_color(imagen):
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    rangos = {
        "verde": ([45, 80, 80], [85, 255, 255]),
        "rojo1": ([0, 80, 80], [10, 255, 255]),
        "rojo2": ([170, 80, 80], [180, 255, 255]),
        "amarillo": ([20, 80, 80], [35, 255, 255])
    }

    centros = {}

    for color, (bajo, alto) in rangos.items():
        mascara = cv2.inRange(hsv, np.array(bajo), np.array(alto))
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contornos = sorted(contornos, key=cv2.contourArea, reverse=True)

        for cnt in contornos:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centros[color] = (cx, cy)
                break

    # Se aceptan ambos tonos de rojo como rojo1 y rojo2
    rojos = []
    if "rojo1" in centros:
        rojos.append(centros["rojo1"])
    if "rojo2" in centros:
        rojos.append(centros["rojo2"])

    # Validación de todos los colores
    if "verde" in centros and len(rojos) == 2 and "amarillo" in centros:
        leds = [centros["verde"], rojos[0], rojos[1], centros["amarillo"]]
        return np.array(leds, dtype=np.float32)
    else:
        return None

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

    led_centers = detectar_leds_por_color(frame)
    if led_centers is not None:
        ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, led_centers, mtx, dist,
                                                        reprojectionError=8.0, confidence=0.99,
                                                        flags=cv2.SOLVEPNP_ITERATIVE)
        if ret:
            # Dibujar LEDs detectados
            colores_bgr = [(0, 255, 0), (0, 0, 255), (0, 0, 180), (0, 255, 255)]  # Verde, Rojo1, Rojo2, Amarillo
            for (x, y), color in zip(led_centers, colores_bgr):
                if np.isfinite(x) and np.isfinite(y):
                    cv2.circle(frame, (int(x), int(y)), 6, color, -1)

            # Puntos proyectados del objeto
            projected_points, _ = cv2.projectPoints(objp, rvecs, tvecs, mtx, dist)
            for pt in projected_points:
                x, y = pt.ravel()
                if np.isfinite(x) and np.isfinite(y):
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

            # Centroide del objeto 3D
            centroide_3d = np.mean(objp, axis=0).reshape(1, 3)
            centroide_camara = (cv2.Rodrigues(rvecs)[0] @ centroide_3d.T + tvecs).T[0]
            centroide_2d, _ = cv2.projectPoints(centroide_3d, rvecs, tvecs, mtx, dist)
            cx, cy = centroide_2d.ravel()
            if np.all(np.isfinite([cx, cy])):
                cv2.circle(frame, (int(cx), int(cy)), 10, (0, 0, 255), -1)
                cv2.putText(frame, "Centro", (int(cx) + 10, int(cy) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Proyección de punto punzón entre los LEDs rojos
            punzon_obj = np.array([[30, 0, 0]], dtype=np.float32)  # Entre Rojo1 y Rojo2
            punzon_camara = (cv2.Rodrigues(rvecs)[0] @ punzon_obj.T + tvecs).T[0]
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
            puntos_guardados.append(centroide_camara.tolist())
            print(f"Punto guardado: {centroide_camara}")
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