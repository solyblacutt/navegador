import cv2
import numpy as np
import os

# EN CONSTRUCCION - archivo general

# --- CONFIGURACIÓN ---
carpeta_imagenes = "pnp/leds"  # nombre de la carpeta con las imágenes

# coordenadas 3d conocidas de las LEDs en el sistema del objeto
objp = np.array([[0, 0, 0],
                 [1, 0, 0],
                 [1, 1, 0],
                 [0, 1, 0]], dtype=np.float32)


# parametros intrinsecos de la camara obtenidos de otra clase calibracion
mtx = np.array([[2992.45904, 0, 1489.42745],
                [0, 2979.52349, 2003.95063],
                [0, 0, 1]], dtype=np.float32)

dist = np.array([0.2433, -1.2952, -0.0025, -0.0020, 2.41], dtype=np.float32)

# creo metodo para detectar leds x contraste
# --- FUNCIÓN PARA DETECTAR LOS LEDS AUTOMÁTICAMENTE ---
def detectar_leds_automaticamente(imagen):
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY)

    # detecta contorno
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]

    # calculo centroides de los contornos
    leds = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            leds.append((cx, cy))

    # Ordenar para mantener consistencia (arriba->abajo, izquierda->derecha)
    # Ordenar consistentemente en sentido antihorario de la camara,  desde el punto superior izquierdo
    # permite estandarizar secuencia de leds
    leds = np.array(leds, dtype=np.float32)
    c = np.mean(leds, axis=0) # calculo el centro geometrico de los leds
    angles = np.arctan2(leds[:, 1] - c[1], leds[:, 0] - c[0]) # calculo angulo [pi,-pi]
    leds = leds[np.argsort(angles)] #organizo leds por su angulo

    return np.array(leds, dtype=np.float32) if len(leds) == 4 else None # condicion tener 4 leds!!!

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
    if not ret:
        print(f"No se pudo resolver PnP en {nombre_archivo}")
        continue
    # proyecto los puntos 3D en las coordenadas 2D de la imagen
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
    cx = cx.item()
    cy = cy.item()

    if not np.all(np.isfinite(centroide_2d)):
        print(f"Centroide proyectado inválido: {centroide_2d}")
        continue

    print(f"Tipo de imagen: {type(imagen)}, shape: {imagen.shape}")
    print(f"Centroide proyectado: cx={cx}, cy={cy}, tipo: {type(cx)}, {type(cy)}")
    print(f"Centroide int: {int(cx)}, {int(cy)}")

    cv2.circle(imagen, (int(cx), int(cy)), 10, (0, 0, 0), -1)
    cv2.putText(imagen, "Centro", (int(cx) + 10, int(cy) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Escalar imagen si es demasiado grande
    max_dim = 1000  # ancho o alto máximo para visualizar
    alto, ancho = imagen.shape[:2]

    # redimensiono la imagen si es que es muy grande
    if max(alto, ancho) > max_dim:
        escala = max_dim / max(alto, ancho)
        imagen_mostrar = cv2.resize(imagen, (int(ancho * escala), int(alto * escala)))
    else:
        imagen_mostrar = imagen

    # Mostrar imagen con proyecciones
    cv2.imshow("Resultado", imagen_mostrar)
    cv2.waitKey(0)

cv2.destroyAllWindows()
