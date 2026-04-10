import cv2
import pickle
import numpy as np

#1. Carga un video con OpenCV con dirección: ./actividad-2-01/video.mp4
cap = cv2.VideoCapture('./actividad-2-01/video.mp4')

def get_frame(cap, n):
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ret, frame = cap.read()
    return frame

#2. Escala la frame 10 a 1.5 veces más grande. Guárdala como frame_10.png
frame = get_frame(cap, 10 - 1)
h, w = frame.shape[:2]
scaled = cv2.resize(frame, (int(w * 1.5), int(h * 1.5)))
cv2.imwrite('frame_10.png', scaled)

# 3. Escala la frame 30 a la mitad de su tamaño. Guárdala como frame_30.png
frame = get_frame(cap, 30 - 1)
h, w = frame.shape[:2]
scaled = cv2.resize(frame, (int(w * 0.5), int(h * 0.5)))
cv2.imwrite('frame_30.png', scaled)

# 4. A la frame 50 rótala 35 grados sobre su centro. Guárdala como frame_50.png
frame = get_frame(cap, 50 - 1)
h, w = frame.shape[:2]
cx, cy = w / 2, h / 2
theta = np.radians(-35)
cos_t, sin_t = np.cos(theta), np.sin(theta)
Rz = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]])
transform_matrix = np.array([[1,0,cx],[0,1,cy],[0,0,1]]) @ Rz @ np.array([[1,0,-cx],[0,1,-cy],[0,0,1]])
M = transform_matrix[:2,:]
rotated = cv2.warpAffine(frame, M, (w, h))
cv2.imwrite('frame_50.png', rotated)

# 5. Guarda la matriz de transformación utilizada en el punto anterior, en su forma estándar, en formato “pickle” como rotation_matrix.pkl
with open('rotation_matrix.pkl', 'wb') as f:
    pickle.dump(transform_matrix, f)
# print("Matriz de rotación guardada en rotation_matrix.pkl")
print(Rz)
print(M)

# 6. A la frame 70 házle un flip horizontal. Guárdala como frame_70.png
frame = get_frame(cap, 70 - 1)
flipped = cv2.flip(frame, 1)
cv2.imwrite('frame_70.png', flipped)

cap.release()
