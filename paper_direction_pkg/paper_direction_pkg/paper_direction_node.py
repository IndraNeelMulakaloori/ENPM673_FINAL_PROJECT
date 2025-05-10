#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import math
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool
from rclpy.qos import qos_profile_sensor_data

class ArucoPoseFollower(Node):
    def __init__(self):
        super().__init__('paper_direction_node')

        self.bridge = CvBridge()
        self.cmd_pub = self.create_publisher(TwistStamped, '/tb4_1/cmd_vel', 10)
        self.create_subscription(CameraInfo, '/tb4_1/oakd/rgb/camera_info', self.camera_info_callback, 10)
        self.create_subscription(Image, '/tb4_1/oakd/rgb/image_raw', self.image_callback, 10)
        self.create_subscription(Bool, '/stop', self.stop_callback, qos_profile_sensor_data)

        self.marker_length = 0.11
        self.dist_coeffs = None
        self.camera_info_received = False

        self.visited_ids = []
        self.pose_counter = 0
        self.required_close_frames = 10
        self.should_stop = False
        self.current_target_id = None

    def camera_info_callback(self, msg):
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)
        self.camera_info_received = True
        self.get_logger().info("Camera calibration received.")

    def image_callback(self, msg):
        if not self.camera_info_received:
            self.get_logger().warn("Waiting for /camera_info...")
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # === PAPER DIRECTION ===
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        _, binary = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(binary, 40, 200, apertureSize=3)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        doc_contour = None
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                doc_contour = approx
                break

        if doc_contour is not None:
            corners = doc_contour.reshape(4, 2)
            center = np.mean(corners, axis=0).astype(int)
            dists = [np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]
            max_idx = np.argmin(dists)
            pt1 = corners[max_idx]
            pt2 = corners[(max_idx + 1) % 4]
            cv2.drawContours(frame, [doc_contour], -1, (0, 255, 0), 3)
            angle_rad = math.atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
            x2 = int(center[0] + 100 * math.cos(angle_rad))
            y2 = int(center[1] + 100 * math.sin(angle_rad))
            cv2.arrowedLine(frame, tuple(center), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, f"Angle: -{math.degrees(angle_rad):.1f} deg", tuple(center),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # === ARUCO DETECTION ===
        found, rvecs, tvecs, ids = self.detect_aruco_pose(gray, frame)

        if not found:
            self.get_logger().info("No ArUco found. Rotating...")
            self.spin_in_place()
        else:
            self.process_aruco_target(ids, tvecs)

        cv2.imshow("Debug View", frame)
        cv2.waitKey(1)

    def detect_aruco_pose(self, gray_img, frame_img):
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(gray_img, aruco_dict, parameters=parameters)

        if ids is not None and len(ids) > 0:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.camera_matrix, self.dist_coeffs)
            cv2.aruco.drawDetectedMarkers(frame_img, corners, ids)
            return True, rvecs, tvecs, ids

        return False, None, None, None

    def process_aruco_target(self, ids, tvecs):
        marker_id = int(ids[0][0])

        if marker_id in self.visited_ids:
            self.get_logger().info(f"Marker {marker_id} already visited. Waiting for new marker.")
            return

        self.current_target_id = marker_id
        tvec = tvecs[0][0]
        self.move_to_marker(tvec, marker_id)

    def move_to_marker(self, tvec, marker_id):
        if self.should_stop:
            self.send_stop_command()
            return

        x = tvec[0]
        z = tvec[2]
        self.get_logger().info(f"[ID {marker_id}] Pose: x={x:.2f} m, z={z:.2f} m")

        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = "base_link"

        close_enough = z < 0.3 and abs(x) < 0.05

        if not close_enough:
            twist_msg.twist.linear.x = 0.15
            twist_msg.twist.angular.z = 0.03 * x
            self.pose_counter = 0
        else:
            self.pose_counter += 1

        if self.pose_counter >= self.required_close_frames:
            self.get_logger().info(f"Reached marker {marker_id}. Stopping and waiting.")
            self.visited_ids.append(marker_id)
            twist_msg.twist.linear.x = 0.0
            twist_msg.twist.angular.z = 0.0
            self.pose_counter = 0

        self.cmd_pub.publish(twist_msg)

    def stop_callback(self, msg):
        self.should_stop = msg.data
        if self.should_stop:
            self.get_logger().info('Received STOP signal')
            self.send_stop_command()
        else:
            self.get_logger().info('Received GO signal')

    def send_stop_command(self):
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = "base_link"
        twist_msg.twist.linear.x = 0.0
        twist_msg.twist.angular.z = 0.0
        self.cmd_pub.publish(twist_msg)

    def spin_in_place(self):
        if self.should_stop:
            self.send_stop_command()
            return

        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = "base_link"
        twist_msg.twist.angular.z = 0.1
        self.cmd_pub.publish(twist_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoPoseFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
