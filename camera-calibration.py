import cv2
import numpy as np
import os

# --- CONFIGURACIÓN ---
carpeta_imagenes = "leds"  # nombre de la carpeta con las imágenes
objp = np.array([[0, 0, 0],
                 [1, 0, 0],
                 [1, 1, 0],
                 [0, 1, 0]], dtype=np.float32)

mtx = np.array([[2992.45904, 0, 1489.42745],
                [0, 2979.52349, 2003.95063],
                [0, 0, 1]], dtype=np.float32)

dist = np.array([0.2433, -1.2952, -0.0025, -0.0020, 2.41], dtype=np.float32)

# --- FUNCIÓN PARA DETECTAR LOS LEDS AUTOMÁTICAMENTE ---
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

    # Ordenar para mantener consistencia (arriba->abajo, izquierda->derecha)
    leds = sorted(leds, key=lambda p: (p[1], p[0]))
    return np.array(leds, dtype=np.float32) if len(leds) == 4 else None

# --- PROCESAR TODAS LAS IMÁGENES EN LA CARPETA ---
for nombre_archivo in os.listdir(carpeta_imagenes):
    if not nombre_archivo.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    ruta = os.path.join(carpeta_imagenes, nombre_archivo)
    imagen = cv2.imread(ruta)
    if imagen is None:
        print(f"No se pudo cargar: {nombre_archivo}")
        continue

    led_centers = detectar_leds_automaticamente(imagen)

    if led_centers is None:
        print(f"No se detectaron 4 LEDs en: {nombre_archivo}")
        continue

    # Resolver PnP y proyectar puntos
    ret, rvecs, tvecs = cv2.solvePnP(objp, led_centers, mtx, dist)
    projected_points, _ = cv2.projectPoints(objp, rvecs, tvecs, mtx, dist)

    for pt in projected_points:
        x, y = pt.ravel()
        cv2.circle(imagen, (int(x), int(y)), 8, (0, 255, 0), -1)
        cv2.putText(imagen, f"{int(x)},{int(y)}", (int(x) + 10, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Dibujar el centroide
    centroide_3d = np.mean(objp, axis=0).reshape(1, 3)
    centroide_2d, _ = cv2.projectPoints(centroide_3d, rvecs, tvecs, mtx, dist)
    cx, cy = centroide_2d.ravel()
    cv2.circle(imagen, (int(cx), int(cy)), 10, (0, 0, 0), -1)
    cv2.putText(imagen, "Centro", (int(cx) + 10, int(cy) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Mostrar imagen con proyecciones
    cv2.imshow("Resultado", imagen)
    cv2.waitKey(0)

cv2.destroyAllWindows()
