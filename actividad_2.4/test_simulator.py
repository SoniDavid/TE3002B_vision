import datetime
import time
import sys
import numpy as np
import cv2

sys.path.append('/home/soni/Documents/classes/IRS_6to/TE3002B_vision/3DGS_Simulator/Interface')
import grpc
import te3002b_pb2
import te3002b_pb2_grpc
import google.protobuf.empty_pb2

from actividad_2_04 import CenterLineDetector


class SimulatorClient:
    def __init__(self, addr='127.0.0.1', port=7072, mode=0):
        self.channel     = grpc.insecure_channel(f'{addr}:{port}')
        self.stub        = te3002b_pb2_grpc.TE3002BSimStub(self.channel)
        self.detector    = CenterLineDetector()
        self.running     = True
        self.timer_delta = 0.025   # 40 Hz
        self.mode        = mode    # 0 = default, 2 = challenge mode

    def configure(self):
        config = te3002b_pb2.ConfigurationData()
        config.resetRobot    = True
        config.mode          = self.mode   # configurable; mode 2 for challenge track
        config.cameraWidth   = 320
        config.cameraHeight  = 240
        config.resetCamera   = False
        config.scene         = 2026
        config.cameraLinear.x  = 0
        config.cameraLinear.y  = 0
        config.cameraLinear.z  = 0
        config.cameraAngular.x = 0
        config.cameraAngular.y = 0
        config.cameraAngular.z = 0
        self.stub.SetConfiguration(config)
        config.resetRobot = False
        time.sleep(0.25)
        self.stub.SetConfiguration(config)

    def _put_text(self, img, text, pos,
                  font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.4, thickness=1):
        """Draw text with a black shadow for readability on any background."""
        x, y = pos
        cv2.putText(img, text, (x + 1, y + 1), font, scale, (0, 0, 0),       thickness)
        cv2.putText(img, text, (x,     y    ), font, scale, (255, 255, 255), thickness)

    def run(self):
        self.configure()

        req = google.protobuf.empty_pb2.Empty()
        cmd = te3002b_pb2.CommandData()

        # ── Video recording setup ────────────────────────────────────────────
        timestamp  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        video_path = f'recording_mode{self.mode}_{timestamp}.avi'
        fourcc     = cv2.VideoWriter_fourcc(*'XVID')
        writer     = cv2.VideoWriter(video_path, fourcc, 40.0, (320, 240))
        print(f"Recording to: {video_path}")

        # FPS counter
        frame_count = 0
        fps_timer   = time.time()
        fps         = 0.0
        angular_z   = 0.0   # initialised here; only updated in mode 0

        # Steering gain for mode 0 (proportional control):
        #   maps ±160 px error → ±0.5 angular_z
        # Mode 2: simulator drives itself — no SetCommand is sent.
        STEERING_GAIN = 1.0 / 320.0   # normalise by fixed camera width

        try:
            while self.running:
                # ── Get camera frame ─────────────────────────────────────────
                result     = self.stub.GetImageFrame(req)
                img_buffer = np.frombuffer(result.data, np.uint8)
                img        = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_LANCZOS4)

                h, w = img.shape[:2]

                # ── Detection ────────────────────────────────────────────────
                cx, cy = self.detector.detect_center_line(img)

                # ── FPS ──────────────────────────────────────────────────────
                frame_count += 1
                now = time.time()
                if now - fps_timer >= 1.0:
                    fps         = frame_count / (now - fps_timer)
                    frame_count = 0
                    fps_timer   = now

                # ── Control ──────────────────────────────────────────────────
                # error: px offset from image centre (always computed for display)
                error = cx - (w // 2)
                # Mode 0: proportional steering sent to simulator
                # Mode 2: simulator drives itself — we only observe, no SetCommand
                if self.mode != 2:
                    angular_z = -float(error) * STEERING_GAIN

                # ── Annotate display frame ────────────────────────────────────
                display_img = img.copy()
                roi_y_vis   = (3 * h) // 4
                CROSS       = 10

                # ROI boundary line (yellow)
                cv2.line(display_img, (0, roi_y_vis), (w, roi_y_vis), (0, 255, 255), 1)

                # Crosshair at detected center (green)
                cv2.line(display_img,
                         (cx - CROSS, cy), (cx + CROSS, cy), (0, 255, 0), 2)
                cv2.line(display_img,
                         (cx, cy - CROSS), (cx, cy + CROSS), (0, 255, 0), 2)

                # Vertical guide line from crosshair down to bottom
                cv2.line(display_img, (cx, cy), (cx, h - 1), (0, 200, 0), 1)

                # Vertical centre reference line (white, thin)
                cv2.line(display_img, (w // 2, roi_y_vis), (w // 2, h - 1),
                         (200, 200, 200), 1)

                # Text overlays (shadowed for readability)
                ctrl_str = 'auto' if self.mode == 2 else f'ang_z={angular_z:+.4f}'
                self._put_text(display_img,
                               f'error={error:+.0f}  {ctrl_str}',
                               (4, 14))
                self._put_text(display_img,
                               f'frame={result.seq}  fps={fps:.1f}',
                               (4, 28))
                self._put_text(display_img,
                               f'mode={self.mode}  cx={cx}  cy={cy}',
                               (4, 42))

                # ── Record and display ────────────────────────────────────────
                writer.write(display_img)
                cv2.imshow('Center Line Detection', display_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break

                # ── Send command (mode 0 only) ────────────────────────────────
                if self.mode != 2:
                    cmd.linear.x  = 0.01
                    cmd.linear.y  = 0.0
                    cmd.linear.z  = 0.0
                    cmd.angular.x = 0.0
                    cmd.angular.y = 0.0
                    cmd.angular.z = angular_z
                    self.stub.SetCommand(cmd)

                time.sleep(self.timer_delta)

        finally:
            writer.release()
            cv2.destroyAllWindows()
            print(f"Recording saved: {video_path}")


def main():
    # Optional positional argument: simulator mode (0 or 2)
    # Usage:  python test_simulator.py        → mode 0
    #         python test_simulator.py 2      → mode 2
    mode   = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    client = SimulatorClient(mode=mode)
    try:
        client.run()
    except KeyboardInterrupt:
        print("Stopped.")
        client.running = False


if __name__ == '__main__':
    main()
