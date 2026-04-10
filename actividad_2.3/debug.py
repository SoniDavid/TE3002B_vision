import numpy as np
import cv2
import os
from actividad_2_03 import OrangeDetector

SAVE_DIR = os.path.join(os.path.dirname(__file__), "debug_frames")
os.makedirs(SAVE_DIR, exist_ok=True)

detector = OrangeDetector()
frame_idx = 0

while True:
    img = detector.capture_image()
    if img is None:
        continue

    # Recompute mask for visualization
    blurred = cv2.GaussianBlur(img, (5, 5), 1.4)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    h_mask = cv2.inRange(h_channel, detector.H_LOW, detector.H_HIGH)
    s_mask = cv2.inRange(s_channel, detector.S_LOW, detector.S_HIGH)
    mask = cv2.bitwise_and(h_mask, s_mask)

    result = detector.detect_orange_object(img)

    display = img.copy()
    if result:
        cx, cy = result
        cv2.circle(display, (cx, cy), 6, (0, 255, 0), -1)
        cv2.line(display, (cx - 12, cy), (cx + 12, cy), (0, 255, 0), 2)
        cv2.line(display, (cx, cy - 12), (cx, cy + 12), (0, 255, 0), 2)
        cv2.putText(display, f"({cx}, {cy})", (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Hue range visualization strip (H: 0-180 full spectrum, detection window highlighted)
    strip_w, strip_h = 360, 60
    hue_strip = np.zeros((strip_h, strip_w, 3), dtype=np.uint8)
    for x in range(strip_w):
        h_val = int(x * 180 / strip_w)
        hue_strip[:, x] = [h_val, 255, 255]
    hue_strip = cv2.cvtColor(hue_strip, cv2.COLOR_HSV2BGR)
    # Highlight detection window with a white rectangle
    x_low  = int(detector.H_LOW  * strip_w / 180)
    x_high = int(detector.H_HIGH * strip_w / 180)
    cv2.rectangle(hue_strip, (x_low, 0), (x_high, strip_h - 1), (255, 255, 255), 2)
    cv2.putText(hue_strip, f"H: {detector.H_LOW}-{detector.H_HIGH}", (x_low, strip_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.imshow("CoppeliaSim - Vision Sensor", display)
    cv2.imshow("Mask (H+S)", mask)
    cv2.imshow("Hue Detection Range", hue_strip)

    # Save every 10th frame
    if frame_idx % 10 == 0:
        cv2.imwrite(os.path.join(SAVE_DIR, f"frame_{frame_idx:04d}_display.png"), display)
        cv2.imwrite(os.path.join(SAVE_DIR, f"frame_{frame_idx:04d}_mask.png"), mask)
        cv2.imwrite(os.path.join(SAVE_DIR, f"frame_{frame_idx:04d}_raw.png"), img)
    frame_idx += 1

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
