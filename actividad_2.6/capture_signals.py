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

BASE_DIR = '/home/soni/Documents/classes/IRS_6to/TE3002B_vision/actividad_2.6/signals/from_static_images'

# key → (class folder name, display label)
CLASSES = {
    ord('1'): ('stop_sign',              'stop_sign'),
    ord('2'): ('worker1_signal',         'worker'),
    ord('3'): ('forward_arrow',          'forward'),
    ord('4'): ('forward_to_left_arrow',  'fwd_left'),
    ord('5'): ('forward_to_right',       'fwd_right'),
    ord('6'): ('give_way_signal',        'give_way'),
}

LIN_STEP = 0.05
ANG_STEP = 0.05

KEY_W     = ord('w')
KEY_S     = ord('s')
KEY_A     = ord('a')
KEY_D     = ord('d')
KEY_R     = ord('r')
KEY_F     = ord('f')
KEY_UP    = 82
KEY_DOWN  = 84
KEY_LEFT  = 81
KEY_RIGHT = 83
KEY_0     = ord('0')
KEY_Q     = ord('q')


class SignalCapturer:
    def __init__(self, addr='127.0.0.1', port=7072, mode=1):
        self.channel     = grpc.insecure_channel(f'{addr}:{port}')
        self.stub        = te3002b_pb2_grpc.TE3002BSimStub(self.channel)
        self.running     = True
        self.timer_delta = 0.025
        self.mode        = mode

        self.cam_x  = self.cam_y  = self.cam_z  = 0.0
        self.cam_rx = self.cam_ry = self.cam_rz = 0.0
        self._config = None

        # counters and flash feedback
        self.counts      = {folder: 0 for folder, _ in CLASSES.values()}
        self.flash_label = ''
        self.flash_until = 0.0

        # create output folders
        for folder, _ in CLASSES.values():
            os.makedirs(os.path.join(BASE_DIR, folder), exist_ok=True)

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

    def _handle_key(self, key, img):
        """Returns False to quit, saves frame if a class key is pressed."""
        if key == KEY_Q:
            return False
        elif key == KEY_W:     self.cam_x += LIN_STEP
        elif key == KEY_S:     self.cam_x -= LIN_STEP
        elif key == KEY_A:     self.cam_y += LIN_STEP
        elif key == KEY_D:     self.cam_y -= LIN_STEP
        elif key == KEY_R:     self.cam_z += LIN_STEP
        elif key == KEY_F:     self.cam_z -= LIN_STEP
        elif key == KEY_UP:    self.cam_ry += ANG_STEP
        elif key == KEY_DOWN:  self.cam_ry -= ANG_STEP
        elif key == KEY_LEFT:  self.cam_rz += ANG_STEP
        elif key == KEY_RIGHT: self.cam_rz -= ANG_STEP
        elif key == KEY_0:
            self.cam_x = self.cam_y = self.cam_z = 0.0
            self.cam_rx = self.cam_ry = self.cam_rz = 0.0
        elif key in CLASSES:
            folder, label = CLASSES[key]
            n      = self.counts[folder]
            fname  = os.path.join(BASE_DIR, folder, f'{folder}_{n:04d}.png')
            cv2.imwrite(fname, img)
            self.counts[folder] += 1
            self.flash_label = f'SAVED  {label}  ({self.counts[folder]})'
            self.flash_until = time.time() + 0.5
            print(f"  saved: {fname}")
        return True

    def _put_text(self, img, text, pos,
                  font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.38, thickness=1,
                  color=(255, 255, 255)):
        x, y = pos
        cv2.putText(img, text, (x + 1, y + 1), font, scale, (0, 0, 0),  thickness)
        cv2.putText(img, text, (x,     y    ), font, scale, color,       thickness)

    def run(self):
        self.configure()

        req         = google.protobuf.empty_pb2.Empty()
        frame_count = 0
        fps_timer   = time.time()
        fps         = 0.0

        print("Signal image capturer — mode", self.mode)
        print("  Move camera : W/S=fwd/back  A/D=strafe  R/F=up/dn  arrows=pitch/yaw  0=reset")
        print("  Capture     : 1=stop  2=worker  3=forward  4=fwd_left  5=fwd_right  6=give_way")
        print("  Quit        : Q")

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

                display = img.copy()

                # HUD
                self._put_text(display, f'fps={fps:.1f}  frame={result.seq}', (4, 13))
                self._put_text(display,
                               f'pos x={self.cam_x:+.2f} y={self.cam_y:+.2f} z={self.cam_z:+.2f}',
                               (4, 26))
                self._put_text(display,
                               f'pitch={self.cam_ry:+.2f}  yaw={self.cam_rz:+.2f}',
                               (4, 39))

                # per-class counters (right side)
                for i, (key, (folder, label)) in enumerate(CLASSES.items()):
                    self._put_text(display,
                                   f'{chr(key)}:{label}={self.counts[folder]}',
                                   (4, 55 + i * 12))

                # key hint strip
                self._put_text(display,
                               'W/S/A/D/R/F:move  arrows:look  0:reset  Q:quit',
                               (4, 232), scale=0.32)

                # flash on capture
                if now < self.flash_until:
                    self._put_text(display, self.flash_label, (60, 122),
                                   scale=0.7, thickness=2, color=(0, 255, 0))

                cv2.imshow('Signal Capturer', display)
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key(key, img):
                    self.running = False
                    break

                self._send_camera_pose()
                time.sleep(self.timer_delta)

        finally:
            cv2.destroyAllWindows()
            print("\nCapture summary:")
            for folder, label in CLASSES.values():
                print(f"  {label:22s}: {self.counts[folder]} images → {BASE_DIR}/{folder}/")


def main():
    mode   = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    client = SignalCapturer(mode=mode)
    try:
        client.run()
    except KeyboardInterrupt:
        print("Stopped.")
        client.running = False


if __name__ == '__main__':
    main()
