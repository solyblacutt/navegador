import cv2
import numpy as np
import os
from pathlib import Path

# ---------------------------
#  CONFIGURACIÓN GENERAL
# ---------------------------
OBJP = np.array([[0, 0, 0],
                 [1, 0, 0],
                 [1, 1, 0],
                 [0, 1, 0]], dtype=np.float32)

CAM_MTX = np.array([[2992.45904, 0, 1489.42745],
                    [0, 2979.52349, 2003.95063],
                    [0, 0, 1]], dtype=np.float32)

CAM_DIST = np.array([0.2433, -1.2952, -0.0025, -0.0020, 2.41], dtype=np.float32)

LED_TXT = Path("led_centers.txt")  # se sobre‑escribe en cada iteración

# ---------------------------
#  CLASE: Segmentador 4 LEDs
# ---------------------------
class Segmentador4LEDs:
    """Detecta 4 LEDs brillantes y escribe sus centros en un .txt"""

    def __init__(self, umbral: int = 220):
        self.umbral = umbral

    @staticmethod
    def _resize(image, max_w=800, max_h=600):
        h, w = image.shape[:2]
        scale = min(max_w / w, max_h / h)
        return cv2.resize(image, (int(w * scale), int(h * scale)))

    def procesar(self, img_path: Path) -> np.ndarray | None:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"❌ No se pudo leer {img_path}")
            return None

        gray = cv2.cvtColor(self._resize(img), cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thr = cv2.threshold(blur, self.umbral, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]

        leds = []
        for c in cnts:
            m = cv2.moments(c)
            if m["m00"] != 0:
                leds.append((int(m["m10"]/m["m00"]), int(m["m01"]/m["m00"])))

        leds = sorted(leds, key=lambda p: (p[1], p[0]))
        if len(leds) != 4:
            print(f"⚠️  No se detectaron 4 LEDs en {img_path}")
            return None

        # escribir txt
        with open(LED_TXT, "w", encoding="utf-8") as f:
            for x, y in leds:
                f.write(f"{x}, {y}\n")

        return np.array(leds, dtype=np.float32)

# ---------------------------
#  CLASE: PnP + Visualización
# ---------------------------
class PnPVisualizador:
    """Resuelve PnP, dibuja proyección y centroide"""

    def __init__(self, mtx: np.ndarray, dist: np.ndarray, objp: np.ndarray, window_name="PnP resultado", interactivo=True):
        self.mtx = mtx
        self.dist = dist
        self.objp = objp
        self.win = window_name
        self.interactivo = interactivo  # True ➜ esperar tecla, False ➜ avance automático
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)

    @staticmethod
    def _resize(image, max_w=800, max_h=600):
        h, w = image.shape[:2]
        scale = min(max_w / w, max_h / h)
        return cv2.resize(image, (int(w * scale), int(h * scale)))

    def procesar(self, img_path: Path, leds_2d: np.ndarray):
        frame = cv2.imread(str(img_path))
        if frame is None or leds_2d is None:
            return

        ok, rvec, tvec = cv2.solvePnP(self.objp, leds_2d, self.mtx, self.dist)
        if not ok:
            print(f"❌ PnP falló en {img_path}")
            return

        proj, _ = cv2.projectPoints(self.objp, rvec, tvec, self.mtx, self.dist)
        proj = proj.reshape(-1, 2).astype(int)

        reproj, _ = cv2.projectPoints(OBJP, rvec, tvec, CAM_MTX, CAM_DIST)
        error = np.linalg.norm(reproj.reshape(-1, 2) - leds_2d, axis=1).mean()

        if error > 10:  # píxeles tolerables
            print("   ⚠️ PnP no fiable (reproj error =", error, "), se omite frame")
            return


        vis = self._resize(frame)
        sx, sy = vis.shape[1] / frame.shape[1], vis.shape[0] / frame.shape[0]
        for (x, y) in proj:
            cx, cy = int(x * sx), int(y * sy)
            cv2.circle(vis, (cx, cy), 6, (0, 255, 0), -1)
            cv2.putText(vis, f"{cx},{cy}", (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # --- CENTROIDE ---
        centroid3d = OBJP.mean(axis=0).reshape(1, 3)
        centroid2d, _ = cv2.projectPoints(centroid3d, rvec, tvec, CAM_MTX, CAM_DIST)

        # 1) achato -> [x, y], 2) paso a float nativo, 3) escalo, 4) int real
        cx_raw, cy_raw = map(float, centroid2d.reshape(-1))
        cx = int(round(cx_raw * sx))
        cy = int(round(cy_raw * sy))
        print("TIPOS:", type(cx), type(cy), cx, cy)

        # límite: ancho y alto de la imagen redimensionada
        h_img, w_img = vis.shape[:2]

        # dibujar solo si está dentro de la ventana y en un rango manejable
        if 0 <= cx < w_img and 0 <= cy < h_img:
            cv2.circle(vis, (cx, cy), 8, (0, 0, 0), -1)
            cv2.putText(vis, "Centro", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        else:
            print("   ⚠️ Centroide fuera de la imagen, se omite:", cx, cy)


        cv2.circle(vis, (cx, cy), 8, (0, 0, 0), -1)
        cv2.putText(vis, "Centro", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imshow(self.win, vis)
        # ---- Espera de tecla ----
        if self.interactivo:
            key = cv2.waitKey(0)  # espera hasta que presiones una tecla
            if key == 27:  # Esc para abortar todo
                raise KeyboardInterrupt
        else:
            cv2.waitKey(2000)  # 0.5 s y avanza


# ---------------------------
#  CLASE: BatchRunner
# ---------------------------
class BatchRunner:
    def __init__(self, carpeta: str, interactivo: bool = True):
        self.folder = Path(carpeta)
        self.segmentador = Segmentador4LEDs()
        self.visual = PnPVisualizador(CAM_MTX, CAM_DIST, OBJP, interactivo=interactivo)

    def run(self):
        try:
            imgs = sorted(self.folder.glob("*.png")) + \
                   sorted(self.folder.glob("*.jpg")) + \
                   sorted(self.folder.glob("*.jpeg"))

            for img_path in imgs:
                print(f"▶ Procesando {img_path.name}")
                leds = self.segmentador.procesar(img_path)
                if leds is not None:
                    self.visual.procesar(img_path, leds)
        except KeyboardInterrupt:
            print("⏹ Proceso interrumpido por el usuario.")
        finally:
            cv2.destroyAllWindows()

# ---------------------------
#  MAIN
# ---------------------------
if __name__ == "__main__":
    BatchRunner("leds", interactivo=True).run()

