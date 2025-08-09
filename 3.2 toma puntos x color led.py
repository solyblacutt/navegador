import cv2
import numpy as np

# este script es una copia modificada de 3-tomo-3-puntos.py

# --- Parámetros de la cámara y objeto --- 20 mm

# Asumimos un cuadrado de 20x20 mm, con este orden:
# Verde (0,0), Rojo (20,0), Rojo (20,20), Amarillo (0,20)
objp = np.array([
    [0, 0, 0],      # LED VERDE
    [150, 0, 0],     # LED VERDE
    [150, 150, 0],    # LED ROJO
    [0, 150, 0]      # LED ROJO
], dtype=np.float32)


mtx = np.array([[1.23994321e+03, 0.00000000e+00, 9.42066048e+02],
 [0.00000000e+00, 1.24162269e+03, 5.16545687e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)

dist = np.array([ 0.02489004,  0.12455246, -0.01055148,  0.00068239,  0.14485304], dtype=np.float32)

# --- Almacenamieto de puntos seleccionados ---
puntos_guardados = []

# Variables globales para tracking temporal
tracking_leds = {
    "verde1": None,
    "verde2": None,
    "azul1": None,
    "azul2": None
}
MAX_DIST_PIXEL = 50  # Distancia máxima permitida frame a frame para tracking


# probar sin esto, da mejor? por el angulo de giro
"""def validar_angulos_cuadrado(pts, tol=10):
    #Valida que los ángulos entre lados consecutivos estén cerca de 90 grados (±tol).

    def angulo(v1, v2):
        cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_ang = np.clip(cos_ang, -1.0, 1.0)
        return np.degrees(np.arccos(cos_ang))

    pts = np.array(pts)
    for i in range(4):
        v1 = pts[(i + 1) % 4] - pts[i]
        v2 = pts[(i - 1) % 4] - pts[i]
        a = angulo(v1, v2)
        if not (90 - tol <= a <= 90 + tol):
            return False
    return True
"""

def validar_escala_cuadrado(pts, escala_mm=150, tol_px=15):
    """Valida escala según distancias entre lados comparadas con la escala esperada."""
    pts = np.array(pts)
    d01 = np.linalg.norm(pts[0] - pts[1])
    d12 = np.linalg.norm(pts[1] - pts[2])
    d23 = np.linalg.norm(pts[2] - pts[3])
    d30 = np.linalg.norm(pts[3] - pts[0])
    distancias = [d01, d12, d23, d30]
    media = np.mean(distancias)

    # Asumiendo escala mm->pixeles depende de escena, ajustar límites medidos
    if any(abs(d - media) > tol_px for d in distancias):
        return False

    # Puedes agregar una verificación de rango total si conoces píxeles por mm.

    return True


# ver bien que hace esto
def actualizar_tracker(nuevos_puntos, colores=["verde1", "verde2", "azul1", "azul2"]):
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
            else:
                pass
    return np.array([tracking_leds[c] for c in colores], dtype=np.float32)



# Función para calcular ROI basado en detección previa
def calcular_roi(centros, margen=50):
    if centros is None:
        return None
    centros = np.array(centros)
    x_min = max(int(np.min(centros[:, 0]) - margen), 0)
    x_max = int(np.max(centros[:, 0]) + margen)
    y_min = max(int(np.min(centros[:, 1]) - margen), 0)
    y_max = int(np.max(centros[:, 1]) + margen)
    return (x_min, x_max, y_min, y_max)


# Modificación en detectar_leds_por_color para usar ROI
def detectar_leds_por_color(imagen, roi=None):
    if roi:
        x_min, x_max, y_min, y_max = roi
        imagen_roi = imagen[y_min:y_max, x_min:x_max]
    else:
        imagen_roi = imagen

    hsv = cv2.cvtColor(imagen_roi, cv2.COLOR_BGR2HSV)
    rangos = {
        "verde1": ([60, 76, 255], [135, 255, 255]),
        "verde2": ([60, 76, 255], [135, 255, 255]),
        "azul1": ([180, 65, 255], [260, 255, 255]),  # rango de azul (ajusta según tus LEDs)
        "azul2": ([180, 65, 255], [260, 255, 255])   # igual que azul1
    }
    centros_verde = []
    centros_azul = []

    for color, (bajo, alto) in rangos.items():
        mascara = cv2.inRange(hsv, np.array(bajo), np.array(alto))
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contornos = sorted(contornos, key=cv2.contourArea, reverse=True)

        for cnt in contornos:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"]) + (x_min if roi else 0)
                cy = int(M["m01"] / M["m00"]) + (y_min if roi else 0)

                if color == "verde":
                    centros_verde.append((cx, cy))
                elif color.startswith("azul"):
                    centros_azul.append((cx, cy))
                # no considers amarillo ya

                # Para tomar solo 2 maximos verdes y rojos:
                if len(centros_verde) >= 2 and len(centros_azul) >= 2:
                    break

    if len(centros_verde) == 2 and len(centros_azul) == 2:
        # Ordenar verdes y rojos para que coincidan en un orden coherente (e.g. por X)
        centros_verde = sorted(centros_verde, key=lambda p: (p[1], p[0]))  # ordenar por Y, luego X
        centros_azul = sorted(centros_azul, key=lambda p: (p[1], p[0]))

        leds = [centros_verde[0], centros_verde[1], centros_azul[0], centros_azul[1]]
        return np.array(leds, dtype=np.float32)
    else:
        return None

def detectar_leds_por_brillo(imagen):
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Crear detector de blobs
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255  # buscar puntos brillantes
    params.minThreshold = 200
    params.maxThreshold = 255
    params.filterByArea = True
    params.minArea = 10
    params.maxArea = 2000
    params.filterByCircularity = False

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(blurred)

    if len(keypoints) >= 4:
        puntos = [kp.pt for kp in keypoints]
        # Ordenar por coordenada y para que coincidan con el objp
        puntos = sorted(puntos, key=lambda x: (x[1], x[0]))  # Y primero, luego X
        puntos = puntos[:4]  # solo los primeros 4
        puntos_ordenados = ordenar_puntos_en_cuadro(puntos)
        if not es_cuadrado_valido(puntos_ordenados, tolerancia_mm=30):  # tolerancia en píxeles
            return None

        return puntos_ordenados

    else:
        return None

# ver si sacar esto
def ordenar_puntos_en_cuadro(puntos):
    if len(puntos) != 4:
        return None

    # Calcular centroide
    centroide = np.mean(puntos, axis=0)

    # Calcular ángulos de cada punto con respecto al centroide
    angulos = []
    for pt in puntos:
        dx, dy = pt[0] - centroide[0], pt[1] - centroide[1]
        angulo = np.arctan2(dy, dx)
        angulos.append((pt, angulo))

    # Ordenar puntos por ángulo (antihorario)
    angulos_ordenados = sorted(angulos, key=lambda x: x[1])
    puntos_ordenados = [pt for pt, _ in angulos_ordenados]

    # Asignar a: [verde, rojo1, rojo2, amarillo]
    # Supongamos que el punto con menor Y es el lado superior del cuadrado
    # Reordenamos para que el primero sea el que esté más arriba a la izquierda

    puntos_ordenados = np.array(puntos_ordenados, dtype=np.float32)

    # Si necesario, rotamos los puntos hasta que el que esté más arriba y más a la izquierda sea el primero
    idx_min = np.argmin(puntos_ordenados[:, 0] + puntos_ordenados[:, 1])  # heurística simple
    puntos_reordenados = np.roll(puntos_ordenados, -idx_min, axis=0)

    return puntos_reordenados


def es_cuadrado_valido(puntos, tolerancia_mm=5.0):
    if len(puntos) != 4:
        return False

    # Distancias entre lados consecutivos
    d01 = np.linalg.norm(puntos[0] - puntos[1])
    d12 = np.linalg.norm(puntos[1] - puntos[2])
    d23 = np.linalg.norm(puntos[2] - puntos[3])
    d30 = np.linalg.norm(puntos[3] - puntos[0])

    distancias = [d01, d12, d23, d30]
    media = np.mean(distancias)
    if any(abs(d - media) > tolerancia_mm for d in distancias):
        return False

    # Validar diagonales
    diag1 = np.linalg.norm(puntos[0] - puntos[2])
    diag2 = np.linalg.norm(puntos[1] - puntos[3])
    if abs(diag1 - diag2) > tolerancia_mm:
        return False

    return True


# --- Captura de video ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

print("Presioná 'g' para guardar el punto actual")
print("Presioná 's' para guardar archivo CSV final")
print("Presioná 'q' para salir")

roi = None  # Inicializamos sin ROI

while True:
    success, frame = cap.read()
    if not success:
        break

    # Usamos roi para acotar búsqueda si está disponible
    led_centers = detectar_leds_por_color(frame, roi=roi)

    #if led_centers is None:
     #   led_centers = detectar_leds_por_brillo(frame)  # podría también modificarse para incluir ROI

    if led_centers is not None:
        # Validaciones geométricas más estrictas
        if not (es_cuadrado_valido(led_centers, tolerancia_mm=10) and
                #validar_angulos_cuadrado(led_centers, tol=15) and
                validar_escala_cuadrado(led_centers, escala_mm=20, tol_px=15)):
            led_centers = None  # Rechazar detección

    if led_centers is not None:
        # Actualizar tracking para suavizar valores y chequear cambios abruptos
        led_centers = actualizar_tracker(led_centers)

        # Actualizar ROI para siguiente frame
        roi = calcular_roi(led_centers, margen=60)

        # Resolver PnP como antes
        success, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, led_centers, mtx, dist,
                                                            reprojectionError=8.0, confidence=0.99,
                                                            flags=cv2.SOLVEPNP_ITERATIVE)

        if success:
            # Dibujar LEDs detectados
            colores_bgr = [(0, 255, 0), (0, 255, 0), (255, 0, 0), (200, 0, 0)]
            # Verde1, Verde2, Azul1, Azul2

            for (x, y), color in zip(led_centers, colores_bgr):
                if np.isfinite(x) and np.isfinite(y):
                    cv2.circle(frame, (int(x), int(y)), 6, color, -1)

            for i, (x, y) in enumerate(led_centers):
                if np.isfinite(x) and np.isfinite(y):
                    cv2.putText(frame, str(i + 1), (int(x) + 5, int(y) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Puntos proyectados del objeto
            projected_points, _ = cv2.projectPoints(objp, rvecs, tvecs, mtx, dist)
            for pt in projected_points:
                coords = pt.ravel()
                if len(coords) >= 2:
                    x, y = float(coords[0]), float(coords[1])
                    if np.isfinite(x) and np.isfinite(y):
                        centro = (int(round(x)), int(round(y)))
                        if centro is not None and len(centro) >= 2:
                            try:
                                x = int(round(float(centro[0])))
                                y = int(round(float(centro[1])))
                                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
                            except Exception as e:
                                # Opcional: imprimir o saltar si algo falla
                                print(f"Error al dibujar círculo: {e}")
                        else:
                            print("Centro inválido para dibujar")

                        cv2.circle(frame, centro, 4, (0, 255, 0), -1)

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
            punzon_obj = np.array([[-50, 10, 0]], dtype=np.float32)  # Entre Rojo1 y Rojo2
            punzon_camara = (cv2.Rodrigues(rvecs)[0] @ punzon_obj.T + tvecs).T[0]
            punzon_2d, _ = cv2.projectPoints(punzon_obj, rvecs, tvecs, mtx, dist)
            px, py = punzon_2d.ravel()
            if np.all(np.isfinite([px, py])):
                # Convertir explícitamente a float y luego a int con round
                try:
                    x_int = int(round(float(px)))
                    y_int = int(round(float(py)))
                except Exception as e:
                    # Si falla, saltar este punto sin dibujar
                    continue

                cv2.circle(frame, (x_int, y_int), 8, (255, 0, 255), -1)
                cv2.putText(frame, f"{x_int}, {y_int}", (x_int + 10, y_int - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # Mostrar video
    cv2.imshow("PnP en vivo", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('g'):
        if led_centers is not None and success:
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