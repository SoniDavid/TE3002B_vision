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

from actividad_2_05 import TrafficLightDetection

STATE_COLORS = {
    "red":    (0,   0,   255),
    "yellow": (0,   220, 220),
    "green":  (0,   200, 0  ),
    "none":   (150, 150, 150),
}

# Step sizes per key press (accumulated into absolute pose)
LIN_STEP = 0.05   # meters per key press
ANG_STEP = 0.05   # radians per key press

KEY_W     = ord('w')
KEY_S     = ord('s')
KEY_A     = ord('a')
KEY_D     = ord('d')
KEY_R     = ord('r')   # up
KEY_F     = ord('f')   # down
KEY_UP    = 82         # pitch up
KEY_DOWN  = 84         # pitch down
KEY_LEFT  = 81         # yaw left
KEY_RIGHT = 83         # yaw right
KEY_0     = ord('0')   # reset camera pose
KEY_Q     = ord('q')   # quit


class SimulatorClient:
    def __init__(self, addr='127.0.0.1', port=7072, mode=1):
        self.channel     = grpc.insecure_channel(f'{addr}:{port}')
        self.stub        = te3002b_pb2_grpc.TE3002BSimStub(self.channel)
        self.detector    = TrafficLightDetection(debug=True)
        self.running     = True
        self.timer_delta = 0.025
        self.mode        = mode

        # accumulated absolute camera pose sent via SetConfiguration each frame
        self.cam_x   = 0.0
        self.cam_y   = 0.0
        self.cam_z   = 0.0
        self.cam_rx  = 0.0   # roll  (unused but available)
        self.cam_ry  = 0.0   # pitch
        self.cam_rz  = 0.0   # yaw

        self._config = None  # reused ConfigurationData object

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
        self._config = cfg

    def _send_camera_pose(self):
        cfg = self._config
        cfg.resetRobot      = False
        cfg.resetCamera     = False
        cfg.cameraLinear.x  = self.cam_x
        cfg.cameraLinear.y  = self.cam_y
        cfg.cameraLinear.z  = self.cam_z
        cfg.cameraAngular.x = self.cam_rx
        cfg.cameraAngular.y = self.cam_ry
        cfg.cameraAngular.z = self.cam_rz
        self.stub.SetConfiguration(cfg)

    def _handle_key(self, key):
        """Update accumulated camera pose from key. Returns False to quit."""
        if key == KEY_Q:
            return False
        elif key == KEY_W:
            self.cam_x += LIN_STEP
        elif key == KEY_S:
            self.cam_x -= LIN_STEP
        elif key == KEY_A:
            self.cam_y += LIN_STEP
        elif key == KEY_D:
            self.cam_y -= LIN_STEP
        elif key == KEY_R:
            self.cam_z += LIN_STEP
        elif key == KEY_F:
            self.cam_z -= LIN_STEP
        elif key == KEY_UP:
            self.cam_ry += ANG_STEP
        elif key == KEY_DOWN:
            self.cam_ry -= ANG_STEP
        elif key == KEY_LEFT:
            self.cam_rz += ANG_STEP
        elif key == KEY_RIGHT:
            self.cam_rz -= ANG_STEP
        elif key == KEY_0:
            self.cam_x = self.cam_y = self.cam_z = 0.0
            self.cam_rx = self.cam_ry = self.cam_rz = 0.0
        return True

    def _put_text(self, img, text, pos,
                  font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.45, thickness=1,
                  color=(255, 255, 255)):
        x, y = pos
        cv2.putText(img, text, (x + 1, y + 1), font, scale, (0, 0, 0), thickness)
        cv2.putText(img, text, (x,     y    ), font, scale, color,      thickness)

    def run(self):
        self.configure()

        req          = google.protobuf.empty_pb2.Empty()
        frame_count  = 0
        fps_timer    = time.time()
        fps          = 0.0

        timestamp    = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        video_path   = f'recording_{timestamp}.avi'
        debug_path   = f'debug_{timestamp}.avi'
        fourcc       = cv2.VideoWriter_fourcc(*'XVID')
        writer       = cv2.VideoWriter(video_path, fourcc, 40.0, (320, 240))
        debug_writer = None
        print(f"Recording to: {video_path}")
        print("Controls: W/S=fwd/back  A/D=strafe  R/F=up/dn  arrows=pitch/yaw  0=reset  Q=quit")

        try:
            while self.running:
                result     = self.stub.GetImageFrame(req)
                img_buffer = np.frombuffer(result.data, np.uint8)
                img        = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_LANCZOS4)

                state = self.detector.detect_state(img)

                frame_count += 1
                now = time.time()
                if now - fps_timer >= 1.0:
                    fps         = frame_count / (now - fps_timer)
                    frame_count = 0
                    fps_timer   = now

                display = img.copy()
                color   = STATE_COLORS.get(state, (150, 150, 150))
                self._put_text(display, f'state: {state}', (4, 16),
                               scale=0.6, thickness=2, color=color)
                self._put_text(display, f'fps={fps:.1f}  frame={result.seq}', (4, 32))
                self._put_text(display,
                               f'pos x={self.cam_x:+.2f} y={self.cam_y:+.2f} z={self.cam_z:+.2f}',
                               (4, 48))
                self._put_text(display,
                               f'rot pitch={self.cam_ry:+.2f} yaw={self.cam_rz:+.2f}',
                               (4, 64))
                self._put_text(display,
                               'W/S:fwd  A/D:strafe  R/F:up/dn  arrows:look  0:reset  Q:quit',
                               (4, 232), scale=0.35)

                writer.write(display)
                cv2.imshow('Traffic Light Detection', display)

                if self.detector.debug and self.detector.debug_frame is not None:
                    dbg = self.detector.debug_frame
                    if debug_writer is None:
                        dh, dw = dbg.shape[:2]
                        debug_writer = cv2.VideoWriter(debug_path, fourcc, 40.0, (dw, dh))
                    debug_writer.write(dbg)
                    cv2.imshow('Pipeline Debug', dbg)

                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key(key):
                    self.running = False
                    break

                # move camera via SetConfiguration (accumulated absolute pose)
                self._send_camera_pose()

                time.sleep(self.timer_delta)

        finally:
            writer.release()
            if debug_writer is not None:
                debug_writer.release()
                print(f"Debug video saved: {debug_path}")
            cv2.destroyAllWindows()
            print(f"Recording saved: {video_path}")


def main():
    mode   = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    client = SimulatorClient(mode=mode)
    try:
        client.run()
    except KeyboardInterrupt:
        print("Stopped.")
        client.running = False


if __name__ == '__main__':
    main()
