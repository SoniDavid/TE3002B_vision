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
    def __init__(self, addr='127.0.0.1', port=7072):
        self.channel = grpc.insecure_channel(f'{addr}:{port}')
        self.stub = te3002b_pb2_grpc.TE3002BSimStub(self.channel)
        self.detector = CenterLineDetector()
        self.running = True
        self.timer_delta = 0.025

    def configure(self):
        config = te3002b_pb2.ConfigurationData()
        config.resetRobot = True
        config.mode = 0
        config.cameraWidth = 320
        config.cameraHeight = 240
        config.resetCamera = False
        config.scene = 2026
        config.cameraLinear.x = 0
        config.cameraLinear.y = 0
        config.cameraLinear.z = 0
        config.cameraAngular.x = 0
        config.cameraAngular.y = 0
        config.cameraAngular.z = 0
        self.stub.SetConfiguration(config)
        config.resetRobot = False
        time.sleep(0.25)
        self.stub.SetConfiguration(config)

    def run(self):
        self.configure()
        req = google.protobuf.empty_pb2.Empty()
        cmd = te3002b_pb2.CommandData()

        while self.running:
            # Get camera frame from simulator
            result = self.stub.GetImageFrame(req)
            img_buffer = np.frombuffer(result.data, np.uint8)
            img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_LANCZOS4)

            # Run center line detection
            cx, cy = self.detector.detect_center_line(img)
            print(f"Frame {result.seq} — center line at ({cx}, {cy})")

            # Draw result on image for visualization
            cv2.circle(img, (cx, cy), 6, (0, 255, 0), -1)
            cv2.line(img, (cx, 0), (cx, img.shape[0]), (0, 255, 0), 1)
            cv2.imshow('Center Line Detection', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

            # Proportional steering based on detected center line position
            error = cx - (img.shape[1] // 2)
            angular_z = -float(error) / img.shape[1]  # normalized to [-0.5, 0.5]

            cmd.linear.x = 0.01
            cmd.linear.y = 0.0
            cmd.linear.z = 0.0
            cmd.angular.x = 0.0
            cmd.angular.y = 0.0
            cmd.angular.z = angular_z
            self.stub.SetCommand(cmd)

            time.sleep(self.timer_delta)

        cv2.destroyAllWindows()


def main():
    client = SimulatorClient()
    try:
        client.run()
    except KeyboardInterrupt:
        print("Stopped.")
        client.running = False
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
