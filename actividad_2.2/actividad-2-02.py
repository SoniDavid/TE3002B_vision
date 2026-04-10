import time
import sys
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import cv2
import numpy as np
import math

client = RemoteAPIClient()
sim = client.getObject('sim')
sim.startSimulation()
time.sleep(2) 

sensor1Handle = sim.getObject('/Vision_sensor')
img, [resX, resY] = sim.getVisionSensorImg(sensor1Handle)
img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
img = cv2.flip(img, 0)
img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

cv2.imwrite('resultado.png', img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

sim.stopSimulation()
