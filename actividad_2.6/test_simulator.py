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

from actividad_2_06 import SignalDetector, CLR_TMPL, CLR_YOLO, CLR_BOTH

# ── Detection colour per source ───────────────────────────────────────────────
SOURCE_COLOR = {'tmpl': CLR_TMPL, 'yolo': CLR_YOLO, 'both': CLR_BOTH}

# ── Camera movement step sizes ────────────────────────────────────────────────
LIN_STEP = 0.05
ANG_STEP = 0.05

KEY_W     = ord('w');  KEY_S  = ord('s')
KEY_A     = ord('a');  KEY_D  = ord('d')
KEY_R     = ord('r');  KEY_F  = ord('f')
KEY_UP    = 82;        KEY_DOWN  = 84
KEY_LEFT  = 81;        KEY_RIGHT = 83
KEY_0     = ord('0');  KEY_Q  = ord('q')


class SimulatorClient:
    def __init__(self, addr='127.0.0.1', port=7072, mode=1):
        self.channel     = grpc.insecure_channel(f'{addr}:{port}')
        self.stub        = te3002b_pb2_grpc.TE3002BSimStub(self.channel)
        self.detector    = SignalDetector(debug=True)
        self.running     = True
        self.timer_delta = 0.025
        self.mode        = mode

        self.cam_x  = self.cam_y  = self.cam_z  = 0.0
        self.cam_rx = self.cam_ry = self.cam_rz = 0.0
        self._config = None

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
        if key == KEY_Q:       return False
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
        return True

    def _put_text(self, img, text, pos,
                  font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.4, thickness=1,
                  color=(255, 255, 255)):
        x, y = pos
        cv2.putText(img, text, (x+1, y+1), font, scale, (0, 0, 0),  thickness)
        cv2.putText(img, text, (x,   y  ), font, scale, color,       thickness)

    def _draw_detections(self, canvas, detections):
        for det in detections:
            color = SOURCE_COLOR.get(det['source'], (200, 200, 200))
            x, y, w, h = det['bbox']
            cv2.rectangle(canvas, (x, y), (x+w, y+h), color, 2)
            label = f"{det['class']} {det['conf']:.2f} [{det['source']}]"
            cv2.putText(canvas, label, (x, max(y-5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2)
            cv2.putText(canvas, label, (x, max(y-5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    def run(self):
        self.configure()

        req         = google.protobuf.empty_pb2.Empty()
        frame_count = 0
        fps_timer   = time.time()
        fps         = 0.0

        timestamp   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        video_path  = f'recording_mode{self.mode}_{timestamp}.avi'
        debug_path  = f'debug_mode{self.mode}_{timestamp}.avi'
        fourcc      = cv2.VideoWriter_fourcc(*'XVID')
        writer      = cv2.VideoWriter(video_path, fourcc, 40.0, (320, 240))
        debug_writer = None
        print(f'Recording to:  {video_path}')
        print(f'Debug video:   {debug_path}')
        print('Move: W/S/A/D/R/F  Look: arrows  Reset: 0  Quit: Q')

        try:
            while self.running:
                result     = self.stub.GetImageFrame(req)
                img_buffer = np.frombuffer(result.data, np.uint8)
                img        = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_LANCZOS4)

                # ── Detection ─────────────────────────────────────────────────
                detections = self.detector.detect(img)

                # ── FPS ───────────────────────────────────────────────────────
                frame_count += 1
                now = time.time()
                if now - fps_timer >= 1.0:
                    fps         = frame_count / (now - fps_timer)
                    frame_count = 0
                    fps_timer   = now

                # ── Display frame ─────────────────────────────────────────────
                display = img.copy()
                self._draw_detections(display, detections)

                self._put_text(display, f'fps={fps:.1f}  frame={result.seq}', (4, 13))
                self._put_text(display,
                               f'pos x={self.cam_x:+.2f} y={self.cam_y:+.2f} z={self.cam_z:+.2f}',
                               (4, 26))
                self._put_text(display,
                               f'pitch={self.cam_ry:+.2f}  yaw={self.cam_rz:+.2f}',
                               (4, 39))
                if not detections:
                    self._put_text(display, 'no detection', (4, 52), color=(100, 100, 100))
                self._put_text(display,
                               'W/S/A/D/R/F:move  arrows:look  0:reset  Q:quit',
                               (4, 232), scale=0.32)

                writer.write(display)
                cv2.imshow('Signal Detection', display)

                # ── Debug pipeline window ─────────────────────────────────────
                if self.detector.debug and self.detector.debug_frame is not None:
                    dbg = self.detector.debug_frame
                    cv2.imshow('Pipeline Debug', dbg)
                    if debug_writer is None:
                        dh, dw = dbg.shape[:2]
                        debug_writer = cv2.VideoWriter(debug_path, fourcc, 40.0, (dw, dh))
                    debug_writer.write(dbg)

                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key(key):
                    self.running = False
                    break

                self._send_camera_pose()
                time.sleep(self.timer_delta)

        finally:
            writer.release()
            if debug_writer is not None:
                debug_writer.release()
                print(f'Debug video saved: {debug_path}')
            cv2.destroyAllWindows()
            print(f'Recording saved: {video_path}')


def main():
    mode   = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    client = SimulatorClient(mode=mode)
    try:
        client.run()
    except KeyboardInterrupt:
        print('Stopped.')
        client.running = False


if __name__ == '__main__':
    main()
