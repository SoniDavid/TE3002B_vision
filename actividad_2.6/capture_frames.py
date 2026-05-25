import datetime
import time
import sys
import os
import numpy as np
import cv2

sys.path.append('/home/soni/Documents/classes/IRS_6to/TE3002B_vision/3DGS_Simulator/Interface')
import grpc
import te3002b_pb2
import te3002b_pb2_grpc
import google.protobuf.empty_pb2

# Save one frame every N simulator frames (1 = every frame)
CAPTURE_EVERY = 5


class FrameCapturer:
    def __init__(self, addr='127.0.0.1', port=7072, mode=2, out_dir=None):
        self.channel     = grpc.insecure_channel(f'{addr}:{port}')
        self.stub        = te3002b_pb2_grpc.TE3002BSimStub(self.channel)
        self.running     = True
        self.timer_delta = 0.025
        self.mode        = mode

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.out_dir = out_dir or f'frames_mode{mode}_{timestamp}'
        os.makedirs(self.out_dir, exist_ok=True)

    def configure(self):
        cfg = te3002b_pb2.ConfigurationData()
        cfg.resetRobot    = True
        cfg.mode          = self.mode
        cfg.cameraWidth   = 320
        cfg.cameraHeight  = 240
        cfg.resetCamera   = False
        cfg.scene         = 2026
        cfg.cameraLinear.x  = 0.0
        cfg.cameraLinear.y  = 0.0
        cfg.cameraLinear.z  = 0.0
        cfg.cameraAngular.x = 0.0
        cfg.cameraAngular.y = 0.0
        cfg.cameraAngular.z = 0.0
        self.stub.SetConfiguration(cfg)
        time.sleep(0.25)
        cfg.resetRobot = False
        self.stub.SetConfiguration(cfg)

    def _put_text(self, img, text, pos,
                  font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.4, thickness=1):
        x, y = pos
        cv2.putText(img, text, (x + 1, y + 1), font, scale, (0, 0, 0),       thickness)
        cv2.putText(img, text, (x,     y    ), font, scale, (255, 255, 255), thickness)

    def run(self):
        self.configure()

        req         = google.protobuf.empty_pb2.Empty()
        frame_count = 0
        saved_count = 0
        fps_timer   = time.time()
        fps         = 0.0

        print(f"Saving frames to: {self.out_dir}/")
        print(f"Capturing every {CAPTURE_EVERY} frame(s). Press Q to stop.")

        try:
            while self.running:
                result     = self.stub.GetImageFrame(req)
                img_buffer = np.frombuffer(result.data, np.uint8)
                img        = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_LANCZOS4)

                frame_count += 1
                now = time.time()
                if now - fps_timer >= 1.0:
                    fps         = frame_count / (now - fps_timer)
                    frame_count = 0
                    fps_timer   = now

                # Save raw frame (no overlays) every CAPTURE_EVERY frames
                if result.seq % CAPTURE_EVERY == 0:
                    fname = os.path.join(self.out_dir, f'frame_{result.seq:06d}.png')
                    cv2.imwrite(fname, img)
                    saved_count += 1

                # Display with overlays (overlays not saved to disk)
                display = img.copy()
                self._put_text(display, f'mode={self.mode} (auto)  fps={fps:.1f}', (4, 14))
                self._put_text(display, f'frame={result.seq}  saved={saved_count}', (4, 28))

                cv2.imshow('Frame Capture', display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break

                time.sleep(self.timer_delta)

        finally:
            cv2.destroyAllWindows()
            print(f"Done. {saved_count} frames saved to {self.out_dir}/")


def main():
    mode    = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    out_dir = sys.argv[2]       if len(sys.argv) > 2 else None
    client  = FrameCapturer(mode=mode, out_dir=out_dir)
    try:
        client.run()
    except KeyboardInterrupt:
        print("Stopped.")
        client.running = False


if __name__ == '__main__':
    main()
