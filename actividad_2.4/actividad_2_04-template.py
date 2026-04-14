import time
import sys
import math
import numpy as np
import cv2

class CenterLineDetector:
    def __init__(self):
        cameraWidth = 320
        cameraHeight = 240

    def detect_center_line(self, image):
        """
        Detecta la línea central de la pista en el 1/4 inferior de la imagen y devuelve las coordenadas del mejor candidato.
        :param img: Imagen en formato OpenCV (BGR).
        :return: Coordenadas del centroide (cx, cy) del mejor candidato en coordenadas de la imagen, None si no se detecta.
        """
        best_candidate = (0, 0)
        return best_candidate
