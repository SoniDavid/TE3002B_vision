import time
import sys
import math
import numpy as np
import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class OrangeDetector:
    def __init__(self):
        """
        Inicializa la conexión con CoppeliaSim y configura el sensor de visión.
        """
        # Conexión con CoppeliaSim
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')

        # Obtener el handle del sensor de visión
        self.sensor1Handle = self.sim.getObject('/Vision_sensor')

    def capture_image(self):
        """
        Captura una imagen desde el sensor de visión de CoppeliaSim.
        :return: Imagen en formato OpenCV (BGR) o None si no se pudo capturar.
        """
        img, [resX, resY] = self.sim.getVisionSensorImg(self.sensor1Handle)
        img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
        img = cv2.flip(img, 0)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img_bgr

    def detect_orange_object(self, img):
        """
        Detecta el objeto naranja en la imagen y devuelve las coordenadas del mejor candidato.
        :param img: Imagen en formato OpenCV (BGR).
        :return: Coordenadas del centroide (cx, cy) del mejor candidato o None si no se detecta.
        """
        # Gaussian blur to reduce noise before thresholding
        blurred = cv2.GaussianBlur(img, (5, 5), 1.4)

        # Convert to HSV and extract H and S channels separately
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]

        # Orange hue range in OpenCV (H: 0-180)
        h_mask = cv2.inRange(h_channel, 5, 25)
        # Saturation threshold: exclude low-sat pixels (floor/walls are near-grey)
        s_mask = cv2.inRange(s_channel, 80, 255)
        mask = cv2.bitwise_and(h_mask, s_mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Best candidate = largest contour by area
        best = max(contours, key=cv2.contourArea)
        moments = cv2.moments(best)

        if moments['m00'] == 0:
            return None

        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        return (cx, cy)

    def run(self):
        """
        Ejecuta el detector en un bucle continuo.
        """
        while True:
            # Capturar la imagen desde el sensor de visión
            img = self.capture_image()

            if img is not None:
                # Detectar el objeto naranja y obtener el mejor candidato
                best_candidate = self.detect_orange_object(img)

                if best_candidate:
                    print(f"Mejor centroide detectado en: {best_candidate}")
                else:
                    print("No se detectó ningún objeto naranja.")


if __name__ == '__main__':
    detector = OrangeDetector()
    detector.run()
