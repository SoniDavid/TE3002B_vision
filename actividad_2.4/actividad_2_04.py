import time
import sys
import math
import numpy as np
import cv2

class CenterLineDetector:
    def __init__(self):
        self.cameraWidth = 320
        self.cameraHeight = 240

    def detect_center_line(self, image):
        """
        Detecta la línea central de la pista en el 1/4 inferior de la imagen y devuelve las coordenadas del mejor candidato.
        :param image: Imagen en formato OpenCV (BGR).
        :return: Coordenadas del centroide (cx, cy) del mejor candidato en coordenadas de la imagen original.
        """
        h, w = image.shape[:2]

        # Crop to the bottom 1/4 of the image
        roi_y = (3 * h) // 4
        roi = image[roi_y:h, :]

        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Gaussian blur to reduce noise (clase 3, slide 33)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

        # Otsu thresholding with THRESH_BINARY_INV so dark track line → white (clase 3, slides 28-30)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find external contours on the binary image (clase 3, slide 44)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return (w // 2, roi_y + (h - roi_y) // 2)

        # Pick the largest contour as the best candidate for the track line
        largest = max(contours, key=cv2.contourArea)

        # Compute centroid via image moments (clase 3, slide 45)
        moments = cv2.moments(largest)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"]) + roi_y  # offset back to original image coords
        else:
            cx = w // 2
            cy = roi_y + (h - roi_y) // 2

        best_candidate = (cx, cy)
        return best_candidate
