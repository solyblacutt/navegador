import cv2
import numpy as np
import os

# EN CONSTRUCCION - archivo general, son las bases del video en loop

# --- CONFIGURACIÓN ---
carpeta_imagenes = "trash/pnp/leds"  # nombre de la carpeta con las imágenes

# coordenadas 3d conocidas de las LEDs en el sistema del objeto
objp = np.array([[0, 0, 0],
                 [2, 0, 0],
                 [2, 2, 0],
                 [0, 2, 0]], dtype=np.float32)


# parametros intrinsecos de la camara de
mtx = np.array([[1.23994321e+03, 0.00000000e+00, 9.42066048e+02],
 [0.00000000e+00, 1.24162269e+03, 5.16545687e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)

dist = np.array([ 0.02489004,  0.12455246, -0.01055148,  0.00068239,  0.14485304], dtype=np.float32)



"""
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



"""

# ---------- rangos HSV de colores ----------
R_H_LOW  = np.array([  0,  80, 120])   # rojo: zona baja   H 0-10
R_H_HIGH = np.array([ 10, 255,255])
R2_H_LOW = np.array([170, 80, 120])    # rojo: zona alta  H 170-180
R2_H_HIGH= np.array([180,255,255])

Y_H_LOW  = np.array([ 20, 80, 120])    # amarillo
Y_H_HIGH = np.array([ 35,255,255])

G_H_LOW  = np.array([ 55, 40, 55])    # verde
G_H_HIGH = np.array([ 85,255,255])

def _clasifica_color(hsv_val):
    h,s,v = hsv_val
    if ((R_H_LOW<=hsv_val).all()  and (hsv_val<=R_H_HIGH).all()) \
    or ((R2_H_LOW<=hsv_val).all() and (hsv_val<=R2_H_HIGH).all()):
        return "rojo"
    if (Y_H_LOW<=hsv_val).all() and (hsv_val<=Y_H_HIGH).all():
        return "amarillo"
    if (G_H_LOW<=hsv_val).all() and (hsv_val<=G_H_HIGH).all():
        return "verde"
    return "otro"

# ---------- detección por contorno + clasificación HSV ----------
def detectar_leds_contorno_color(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Otsu para binarizar las zonas brillantes
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # limpiar ruido
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)

    cnts,_ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts   = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]

    if len(cnts) != 4:
        return None          # detección incompleta

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    leds_dict = {"verde":[], "amarillo":[], "rojo":[]}

    for c in cnts:
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx = int(m["m10"]/m["m00"])
        cy = int(m["m01"]/m["m00"])
        color = _clasifica_color(hsv[cy, cx])
        if color in leds_dict:
            leds_dict[color].append((cx, cy))

    # necesitamos 1 verde, 1 amarillo y 2 rojos
    if not (len(leds_dict["verde"])==1 and len(leds_dict["amarillo"])==1 and len(leds_dict["rojo"])==2):
        return None

    # ordenar los 2 rojos de izquierda a derecha
    leds_dict["rojo"].sort(key=lambda p: p[0])

    # construir orden final:  verde → amarillo → rojo izq → rojo der
    orden = leds_dict["verde"] + leds_dict["amarillo"] + leds_dict["rojo"]
    return np.array(orden, dtype=np.float32)


# --- PROCESAR TODAS LAS IMÁGENES EN LA CARPETA ---
for nombre_archivo in os.listdir(carpeta_imagenes):
    if not nombre_archivo.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    ruta = os.path.join(carpeta_imagenes, nombre_archivo)
    imagen = cv2.imread(ruta)
    if imagen is None:
        print(f"No se pudo cargar: {nombre_archivo}")
        continue

    led_centers = detectar_leds_contorno_color(imagen)

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

    # --- DIBUJAR EL PUNTO (-5, 1, 0) EN EL SISTEMA DEL OBJETO ---
    punto_punzon_obj = np.array([[-5, 1, 0]], dtype=np.float32)
    punto_punzon_2d, _ = cv2.projectPoints(punto_punzon_obj, rvecs, tvecs, mtx, dist)

    x_punzon, y_punzon = punto_punzon_2d.ravel()
    x_punzon = x_punzon.item()
    y_punzon = y_punzon.item()

    cv2.circle(imagen, (int(x_punzon), int(y_punzon)), 10, (255, 0, 255), -1)
    cv2.putText(imagen, f"{int(x_punzon)}, {int(y_punzon)}", (int(x_punzon) + 10, int(y_punzon) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

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
