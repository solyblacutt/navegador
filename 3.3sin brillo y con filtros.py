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

# Variables para tracking temporal
tracking_leds = None  # almacenará np.array con 4 puntos suavizados
MAX_DIST_PIXEL = 50  # máximo desplazamiento permitido frame a frame para tracking

# ROI dinámica
roi = None
ROI_MARGIN = 60  # margen píxeles alrededor de detección previa para ROI

"""
def validar_cuadrado(puntos, tolerancia_dist=15, tolerancia_ang=15):
    
    #Revisa si 4 puntos forman un cuadrado aceptable:
    #- Lados similares (tolerancia en pixeles)
    #- Diagonales similares
    #- Ángulos interiores cercanos a 90° (±tolerancia_ang grados)
    
    if len(puntos) != 4:
        return False

    pts = np.array(puntos)

    # Distancias entre vértices consecutivos (en orden)
    sides = [np.linalg.norm(pts[i] - pts[(i+1) % 4]) for i in range(4)]
    mean_side = np.mean(sides)

    if any(abs(s - mean_side) > tolerancia_dist for s in sides):
        return False

    # Distancias entre diagonales
    d1 = np.linalg.norm(pts[0] - pts[2])
    d2 = np.linalg.norm(pts[1] - pts[3])
    if abs(d1 - d2) > tolerancia_dist:
        return False

    # Ángulos interiores
    def angulo_entre(v1, v2):
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))

    for i in range(4):
        v1 = pts[(i-1) % 4] - pts[i]
        v2 = pts[(i+1) % 4] - pts[i]
        angle = angulo_entre(v1, v2)
        if not (90 - tolerancia_ang <= angle <= 90 + tolerancia_ang):
            return False

    return True
"""
"""
def actualizar_tracking(nuevos_puntos):
    
    #Actualiza el tracking de LEDs para suavizar posiciones y filtrar saltos bruscos.
    
    global tracking_leds

    nuevos = np.array(nuevos_puntos, dtype=np.float32)

    if tracking_leds is None:
        tracking_leds = nuevos
    else:
        distancias = np.linalg.norm(tracking_leds - nuevos, axis=1)
        # Si alguna distancia es muy grande (punto saltó), mantenemos anterior
        tracking_leds = np.where(distancias[:, None] < MAX_DIST_PIXEL,
                                 0.6 * tracking_leds + 0.4 * nuevos,
                                 tracking_leds)
    return tracking_leds
"""

def ordenar_puntos_clockwise(puntos):
    
    #Ordena los 4 puntos en sentido horario alrededor de su centroide,
    #con el primer punto siendo el que está más arriba-izquierda (sum menor).
    
    c = np.mean(puntos, axis=0)
    angles = np.arctan2(puntos[:, 1] - c[1], puntos[:, 0] - c[0])
    idx_sort = np.argsort(angles)
    pts_sorted = puntos[idx_sort]

    # Rotar para que el punto con la menor suma de coords sea el primero
    idx_min = np.argmin(pts_sorted[:, 0] + pts_sorted[:, 1])
    pts_sorted = np.roll(pts_sorted, -idx_min, axis=0)
    return pts_sorted

"""
def calcular_roi(puntos, margen=ROI_MARGIN):
    
    #Calcula los límites de ROI para buscar LEDs, con margen en píxeles.
    
    x_min = max(int(np.min(puntos[:, 0]) - margen), 0)
    x_max = int(np.max(puntos[:, 0]) + margen)
    y_min = max(int(np.min(puntos[:, 1]) - margen), 0)
    y_max = int(np.max(puntos[:, 1]) + margen)

    return x_min, x_max, y_min, y_max
"""

def detectar_leds_automaticamente(imagen, roi=None):
    """
    Detectar LEDs basado en brillo blanco alto (threshold).
    Puede usar ROI para acelerar búsqueda.
    """
    if roi:
        x_min, x_max, y_min, y_max = roi
        img_roi = imagen[y_min:y_max, x_min:x_max]
    else:
        img_roi = imagen

    gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
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
            # Ajuste coords a imagen original si ROI
            if roi:
                cx += x_min
                cy += y_min
            leds.append((cx, cy))

    if len(leds) != 4:
        return None

    leds = np.array(leds, dtype=np.float32)
    leds = ordenar_puntos_clockwise(leds)
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

    # Detectar LEDs usando ROI si existe
    led_centers = detectar_leds_automaticamente(frame, roi)

    # Validar y filtrar detecciones geométricas
    if led_centers is not None:
        #if not validar_cuadrado(led_centers, tolerancia_dist=15, tolerancia_ang=15):
            led_centers = None

    # Actualizar tracking si la detección es válida
    if led_centers is not None:
        #led_centers = actualizar_tracking(led_centers)

        # Actualizar ROI para próximo frame
        #roi = calcular_roi(led_centers)

        # Resolver PnP con RANSAC
        ret_pnp, rvecs, tvecs, inliers = cv2.solvePnPRansac(
            objp, led_centers, mtx, dist,
            reprojectionError=8.0, confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if ret_pnp and inliers is not None and len(inliers) == 4:
            # Dibujar LED detectados
            for (x, y) in led_centers:
                if np.isfinite(x) and np.isfinite(y):
                    cv2.circle(frame, (int(round(x)), int(round(y))), 6, (0, 255, 0), -1)

            # Dibujar puntos proyectados
            projected_points, _ = cv2.projectPoints(objp, rvecs, tvecs, mtx, dist)
            for pt in projected_points:
                x, y = pt.ravel()
                if np.isfinite(x) and np.isfinite(y):
                    cv2.circle(frame, (int(round(x)), int(round(y))), 6, (0, 0, 255), 2)

            # Centroide 3D
            centroide_3d = np.mean(objp, axis=0).reshape(1, 3)
            centroide_2d, _ = cv2.projectPoints(centroide_3d, rvecs, tvecs, mtx, dist)
            cx, cy = centroide_2d.ravel()
            if np.all(np.isfinite([cx, cy])):
                cv2.circle(frame, (int(round(cx)), int(round(cy))), 10, (255, 0, 0), -1)
                cv2.putText(frame, "Centro", (int(round(cx)) + 10, int(round(cy)) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Mostrar video
    cv2.imshow("PnP en vivo", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('g'):
        if led_centers is not None and ret_pnp:
            puntos_guardados.append(centroide_3d.tolist())
            print(f"Punto guardado: {centroide_3d}")
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
