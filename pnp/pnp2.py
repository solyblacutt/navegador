import cv2
import numpy as np

led_centers = []
objp = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]

with open("../led_centers.txt", "r") as f:
    for line in f:
        x, y = map(int, line.strip().split(","))
        led_centers.append((x, y))

# Convertir a NumPy arrays con tipo float32
objp = np.array(objp, dtype=np.float32)
led_centers = np.array(led_centers, dtype=np.float32)

# Matriz de cámara
mtx = np.array([[2.99245904e+03, 0.00000000e+00, 1.48942745e+03],
                [0.00000000e+00, 2.97952349e+03, 2.00395063e+03],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)

# Distorsión
dist = np.array([2.43302187e-01, -1.29527262e+00, -2.50888650e-03,
                 -2.07908157e-03, 2.41003838e+00], dtype=np.float32)

# solvePnP requiere puntos 3D (objp) y 2D (led_centers)
ret, rvecs, tvecs = cv2.solvePnP(objp, led_centers, mtx, dist)

print("rvecs:\n", rvecs)
print("tvecs:\n", tvecs)


"""""
rvecs:
 [[1.72797138]
 [1.81628202]
 [2.26308698]]
tvecs:
 [[-20.38184396]
 [-25.25523136]
 [ 47.66751124]]
 
PNP resuelve la pose de la cámara: es decir, te dice dónde está la cámara y cómo está orientada respecto a un objeto cuyo modelo 3D (objp) ya conocés.
"""

# Proyectar puntos 3D a 2D usando los vectores obtenidos
projected_points, _ = cv2.projectPoints(objp, rvecs, tvecs, mtx, dist)

# Mostrar los puntos proyectados
print("Puntos proyectados (objp transformado con rvecs/tvecs):")
for i, pt in enumerate(projected_points):
    x, y = pt.ravel()
    print(f"Punto {i}: ({x:.2f}, {y:.2f})")

""""
Punto 0: (106.01, 297.25)
Punto 1: (109.88, 364.61)
Punto 2: (181.18, 369.52)
Punto 3: (179.93, 304.62)
"""


def resize_image(image, max_width=800, max_height=600):
    height, width = image.shape[:2]
    scaling_factor = min(max_width / width, max_height / height)
    return cv2.resize(image, (int(width * scaling_factor), int(height * scaling_factor)))


image = cv2.imread('test4leds.jpeg')  # Replace with your image path

# Resize the image to fit the display window
img = resize_image(image)


# Asegurate de tener los puntos proyectados de esta forma:
# projected_points: resultado de cv2.projectPoints
# Esto es un array shape (N,1,2) que hay que convertir a (x, y)
for point in projected_points:
    x, y = point.ravel()
    center = (int(x), int(y))
    cv2.circle(img, center, 10, (0, 255, 0), -1)  # círculo verde
    cv2.putText(img, f"{int(x)},{int(y)}", (center[0] + 10, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


# ---------- Calcular y proyectar el centroide 3D ----------
centroide_3d = np.mean(objp, axis=0).reshape(1, 3)  # shape (1, 3)
centroide_2d, _ = cv2.projectPoints(centroide_3d, rvecs, tvecs, mtx, dist)
x_c, y_c = centroide_2d.ravel()

# Dibujar el centro proyectado
cv2.circle(img, (int(x_c), int(y_c)), 12, (0, 0, 0), -1)  # círculo rojo
cv2.putText(img, "Centro", (int(x_c) + 10, int(y_c) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# Mostrar imagen final
cv2.imshow("Proyección de puntos 3D + Centro", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""""
# Mostrar o guardar imagen con los puntos
cv2.imshow("Proyección de puntos 3D", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""