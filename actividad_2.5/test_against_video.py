"""
Test TrafficLightDetection against the graded reference video A01571777.mp4.
Parses the reference state from the overlaid text by probing pixel brightness
in the text band, then runs the new detector and computes the score.
"""

import sys
import numpy as np
import cv2

sys.path.insert(0, '/home/soni/Documents/classes/IRS_6to/TE3002B_vision/actividad_2.5')
from actividad_2_05 import TrafficLightDetection

VIDEO = '/home/soni/Documents/classes/IRS_6to/TE3002B_vision/actividad_2.5/A01571777.mp4'

SCORE_HIT  =  10.0
SCORE_MISS = -25.0

# Calibrated from probing known frames:
#   "red"    → last bright col ≈ 130  (cols 100-130)
#   "none"   → last bright col ≈ 140
#   "green"  → last bright col ≈ 148
#   "yellow" → last bright col ≈ 153
TEXT_BAND_Y = (41, 53)   # row slice where "Reference: XXXX" lives
TEXT_SCAN_X = (100, 170) # columns after "Reference: "


def parse_reference(frame_gray):
    """Return the reference state string from the text overlay."""
    band = frame_gray[TEXT_BAND_Y[0]:TEXT_BAND_Y[1], TEXT_SCAN_X[0]:TEXT_SCAN_X[1]].max(axis=0)
    bright = np.where(band > 180)[0]
    if len(bright) == 0:
        return "none"
    last = bright[-1]
    # offsets relative to TEXT_SCAN_X[0]
    if last < 35:    # ~col 135 absolute
        return "red"
    elif last < 44:  # ~col 144 absolute
        return "none"
    elif last < 52:  # ~col 152 absolute
        return "green"
    else:
        return "yellow"


def run_test():
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"ERROR: cannot open {VIDEO}")
        return

    detector = TrafficLightDetection(debug=False)

    score   = 0.0
    total   = 0
    correct = 0
    counts  = {}   # (reference, detected) → count

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ref   = parse_reference(gray)
        det   = detector.detect_state(frame)

        total += 1
        hit = (ref == det)
        if hit:
            correct += 1
            score  += SCORE_HIT
        else:
            score  += SCORE_MISS

        counts[(ref, det)] = counts.get((ref, det), 0) + 1

    cap.release()

    accuracy = correct / total * 100 if total else 0
    print(f"\n{'='*50}")
    print(f"Frames processed : {total}")
    print(f"Correct          : {correct}  ({accuracy:.1f}%)")
    print(f"Final score      : {score:.1f}")
    print(f"\nConfusion matrix (ref → detected):")
    all_states = sorted({s for pair in counts for s in pair})
    col_header = "ref\\det"
    header = f"{col_header:<10}" + "".join(f"{s:>10}" for s in all_states)
    print(header)
    for ref in all_states:
        row = f"{ref:<10}" + "".join(
            f"{counts.get((ref, det), 0):>10}" for det in all_states
        )
        print(row)
    print('='*50)


if __name__ == '__main__':
    run_test()
