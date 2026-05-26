import datetime
import time
import sys
import os
import cv2

CAMERA_INDEX = 2
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
TARGET_FPS   = 30


def put_text(img, text, pos,
             font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.5, thickness=1,
             color=(255, 255, 255)):
    x, y = pos
    cv2.putText(img, text, (x + 1, y + 1), font, scale, (0, 0, 0), thickness)
    cv2.putText(img, text, (x,     y    ), font, scale, color,       thickness)


def main():
    camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else CAMERA_INDEX

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: could not open camera {camera_index}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)

    actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS

    timestamp  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path   = f'recording_{timestamp}.mp4'
    fourcc     = cv2.VideoWriter_fourcc(*'mp4v')
    writer     = cv2.VideoWriter(out_path, fourcc, actual_fps, (actual_w, actual_h))

    print(f"Camera {camera_index}: {actual_w}x{actual_h} @ {actual_fps:.0f} fps")
    print(f"Recording to: {out_path}")
    print("Press Q to stop.")

    frame_count = 0
    fps_timer   = time.time()
    fps         = 0.0
    recording   = True

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: failed to grab frame.")
                continue

            if recording:
                writer.write(frame)

            frame_count += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                fps         = frame_count / (now - fps_timer)
                frame_count = 0
                fps_timer   = now

            display = frame.copy()
            status  = "REC" if recording else "PAUSED"
            color   = (0, 0, 255) if recording else (0, 165, 255)
            put_text(display, f'{status}  fps={fps:.1f}', (8, 22), color=color)
            put_text(display, f'{actual_w}x{actual_h}  cam={camera_index}', (8, 42))
            put_text(display, out_path, (8, actual_h - 10), scale=0.38)
            put_text(display, 'P:pause/resume  Q:quit', (8, actual_h - 24), scale=0.38)

            cv2.imshow('USB Camera', display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('p'):
                recording = not recording
                state = "resumed" if recording else "paused"
                print(f"Recording {state}.")

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
