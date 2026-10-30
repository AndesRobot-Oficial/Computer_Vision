import rclpy
from rclpy.node import Node

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time

from geometry_msgs.msg import Point
from std_msgs.msg import String, Int32


class RealSenseYOLONode(Node):

    def __init__(self):
        super().__init__("realsense_yolo_node")

        # Publishers
        self.flag_pub = self.create_publisher(Int32, "yolo_detection_flag", 10)
        self.pub_object = self.create_publisher(String, "detected_object", 10)
        self.pub_point = self.create_publisher(Point, "object_position", 10)

        # Parámetro del modelo
        self.declare_parameter("model_path", "Models/best.onnx")
        model_path = self.get_parameter("model_path").value

        # Cargar modelo
        self.get_logger().info("Cargando modelo YOLO…")
        self.model = YOLO(model_path)
        self.class_names = self.model.names

        # Inicializar RealSense
        self.pipeline, self.align, self.intrinsics = self.init_realsense()

        # Timer
        self.create_timer(0.03, self.process_frame)  # ~30 FPS

    def init_realsense(self):
        self.get_logger().info("Inicializando RealSense D435i…")

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        time.sleep(1)
        return pipeline, align, intr

    def get_3d_point(self, depth_frame, x, y):
        depth_m = depth_frame.get_distance(x, y)

        if depth_m <= 0.01:
            return None, None, None

        X, Y, Z = rs.rs2_deproject_pixel_to_point(
            self.intrinsics, [x, y], depth_m
        )

        return X, Y, Z

    def process_frame(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        except RuntimeError:
            self.get_logger().warn("Frame drop")
            return

        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        if not depth_frame or not color_frame:
            return

        img = np.asanyarray(color_frame.get_data())
        detected = False

        # YOLO
        results = self.model(img, verbose=False)
        CONF_TH = 0.75  # 75 %

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0].cpu().numpy())
                if conf < CONF_TH:
                    continue

                detected = True

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                class_id = int(box.cls[0].cpu().numpy())
                label = self.class_names[class_id]

                # Punto central
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                X, Y, Z = self.get_3d_point(depth_frame, cx, cy)

                # Publicar objeto
                msg_obj = String()
                msg_obj.data = label
                self.pub_object.publish(msg_obj)

                # Publicar punto 3D
                pt = Point()
                pt.x = float(X) if X else 0.0
                pt.y = float(Y) if Y else 0.0
                pt.z = float(Z) if Z else 0.0
                self.pub_point.publish(pt)

                # Debug
                self.get_logger().info(
                    f"Objeto: {label} | Conf: {conf:.2f} | X={pt.x:.2f}, Y={pt.y:.2f}, Z={pt.z:.2f}"
                )

                # Dibujar
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(
                    img,
                    f"{label} {conf:.2f} ({pt.z:.2f}m)",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

        # Publicar bandera
        flag_msg = Int32()
        flag_msg.data = 1 if detected else 0
        self.flag_pub.publish(flag_msg)

        cv2.imshow("ROS2 YOLO + RealSense", img)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = RealSenseYOLONode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.pipeline.stop()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

