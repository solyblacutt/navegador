import cv2

# Reemplaza con la IP y puerto que muestra la app DroidCam en tu iPhone
ip = "192.168.0.91"  # Ejemplo: 192.168.1.139
port = "4747"       # Puerto por defecto que usa DroidCam

# URL para video en formato MJPEG proporcionado por DroidCam
url = f"http://{ip}:{port}/video"

# Captura el video vía URL
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("No se pudo abrir la cámara DroidCam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se recibió frame de video")
        break

    # Muestra el video recibido
    cv2.imshow("DroidCam iPhone", frame)

    # Presiona 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
