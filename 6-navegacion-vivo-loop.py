#!/usr/bin/env python3
"""
Visualizador en tiempo real de la posición del instrumental médico
usando el patrón de 4 LEDs y un modelo STL del fémur.
"""
import cv2
import numpy as np
import open3d as o3d
import time
import os

# ------------------- Configuración -------------------
# Dimensión física del patrón (mm)
PATTERN_SIZE_MM = 40.0

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

# Cuadrado objeto (en mm)
objp = (np.array([[0, 0, 0],
                  [1, 0, 0],
                  [1, 1, 0],
                  [0, 1, 0]], dtype=np.float32) * PATTERN_SIZE_MM)

# Ruta de archivos
FEMUR_STL = "femur.stl"
TRANSFORM_FILE = "cam_to_model.npy"

# ------------------- Utilidades -------------------
def detectar_leds_automaticamente(imagen):
    """Devuelve los 4 centros de LED ordenados horario o None."""
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

    c = np.mean(leds, axis=0)
    angles = np.arctan2(leds[:, 1] - c[1], leds[:, 0] - c[0])
    leds = leds[np.argsort(angles)]
    return leds

def build_homogeneous(R, t):
    """Convierte R (3×3) y t (3,) en matriz 4×4."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T

# ------------------- Cargar modelo 3D -------------------
mesh = o3d.io.read_triangle_mesh(FEMUR_STL)
mesh.compute_vertex_normals()
mesh.paint_uniform_color([0.8, 0.8, 0.8])

# Cargar transformación cámara→modelo (4×4)
if not os.path.exists(TRANSFORM_FILE):
    raise FileNotFoundError(
        f"No se encontró '{TRANSFORM_FILE}'. Ejecuta registro_3D.py antes."
    )
T_cam_to_model = np.load(TRANSFORM_FILE)

# ------------------- Visualizador -------------------
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Navegación Quirúrgica 3D", width=1280, height=720)
vis.add_geometry(mesh)

# Marco del instrumento (se actualizará cada frame)
instrument_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
vis.add_geometry(instrument_frame)
vis.poll_events()
vis.update_renderer()

prev_T_model = np.eye(4)

# ------------------- Captura de vídeo -------------------
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(URL)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara")

print("Presiona 'q' en la ventana de vídeo para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    led_centers = detectar_leds_automaticamente(frame)
    if led_centers is not None:
        success, rvec, tvec, inliners = cv2.solvePnPRansac(objp, led_centers, mtx, dist,
                                                 flags=cv2.SOLVEPNP_ITERATIVE)
        if success:
            R_cam_obj, _ = cv2.Rodrigues(rvec)
            T_cam_obj = build_homogeneous(R_cam_obj, tvec.flatten())

            # Transformación de objeto (instrumento) al modelo (fémur)
            T_model_obj = T_cam_to_model @ T_cam_obj

            # Actualiza la posición del frame del instrumento
            delta = T_model_obj @ np.linalg.inv(prev_T_model)
            instrument_frame.transform(delta)
            prev_T_model = T_model_obj

            vis.update_geometry(instrument_frame)

            # Dibuja LEDs en la imagen para feedback visual
            for pt in led_centers:
                cv2.circle(frame, tuple(int(x) for x in pt), 6, (0, 255, 0), -1)

    vis.poll_events()
    vis.update_renderer()

    cv2.imshow("Detección de LEDs", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpieza
cap.release()
cv2.destroyAllWindows()
vis.destroy_window()