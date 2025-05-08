#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import math
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from rclpy.qos import qos_profile_sensor_data

class ArucoPoseFollower(Node):
    def __init__(self):
        super().__init__('paper_direction_node')

        self.bridge = CvBridge()
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(CameraInfo, '/camera/camera_info', self.camera_info_callback, 10)
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.create_subscription(Bool, '/stop', self.stop_callback, qos_profile_sensor_data)

        self.marker_length = 0.11
        self.dist_coeffs = None
        self.camera_info_received = False
        self.max_markers = 7
        self.done_moving = False
        self.visited_ids = []
        self.required_close_frames = 10
        self.pose_counter = 0
        self.current_target_id = None
        self.should_stop = False

        # Optical flow
        self.prev_gray = None
        self.motion_threshold = 1.5
        self.obstacle_present = False

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

        # === OPTICAL FLOW: detect dynamic obstacle ===
        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_level = np.mean(mag)

            if motion_level > self.motion_threshold:
                if not self.obstacle_present:
                    self.get_logger().warn(f"Obstacle detected! Motion={motion_level:.2f}")
                self.obstacle_present = True
            else:
                if self.obstacle_present:
                    self.get_logger().info("Obstacle cleared.")
                self.obstacle_present = False

        self.prev_gray = gray.copy()

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
            cv2.putText(frame, f"Angle: {math.degrees(angle_rad):.1f} deg", tuple(center),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # === ARUCO DETECTION ===
        found, rvecs, tvecs, ids = self.detect_aruco_pose(gray, frame)

        if not found:
            self.get_logger().info("No ArUco found. Rotating...")
            self.spin_in_place()
        else:
            self.process_aruco_target(ids, tvecs)

        if self.obstacle_present:
            cv2.putText(frame, "Obstacle Detected", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

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
        best_idx = None
        min_z = float('inf')
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in self.visited_ids:
                continue
            z = tvecs[i][0][2]
            if z < min_z:
                min_z = z
                best_idx = i

        if best_idx is None:
            self.get_logger().info("All ArUco markers already visited.")
            self.spin_in_place()
            return

        self.current_target_id = int(ids[best_idx][0])
        self.move_to_marker(tvecs[best_idx][0], self.current_target_id)

        if len(self.visited_ids) >= self.max_markers and not self.done_moving:
            self.get_logger().info("7 ArUco markers visited. Moving forward and stopping.")
            self.move_forward_and_stop()
            self.done_moving = True

    def move_to_marker(self, tvec, marker_id):
        if self.obstacle_present:
            self.send_stop_command()
            return

        x = tvec[0]
        z = tvec[2]
        self.get_logger().info(f"[ID {marker_id}] Pose: x={x:.2f} m, z={z:.2f} m")

        twist = Twist()
        close_enough = z < 0.3 and abs(x) < 0.05

        if not close_enough:
            twist.linear.x = 0.15
            twist.angular.z = 0.03 * x
            self.pose_counter = 0
        else:
            self.pose_counter += 1

        if self.pose_counter >= self.required_close_frames:
            self.get_logger().info(f"Reached marker {marker_id}. Stopping.")
            self.visited_ids.append(marker_id)
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.pose_counter = 0

        self.cmd_pub.publish(twist)

    def stop_callback(self, msg):
        self.should_stop = msg.data
        if self.should_stop:
            self.get_logger().info('Received STOP signal')
            self.send_stop_command()
        else:
            self.get_logger().info('Received GO signal')

    def send_stop_command(self):
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = 0.0
        self.cmd_pub.publish(twist_msg)

    def move_forward_and_stop(self):
        if self.obstacle_present:
            self.send_stop_command()
            return
        move_twist = Twist()
        move_twist.linear.x = 0.15
        self.cmd_pub.publish(move_twist)
        rclpy.spin_once(self, timeout_sec=2.0)
        self.send_stop_command()

    def spin_in_place(self):
        if self.obstacle_present:
            self.send_stop_command()
            return
        twist = Twist()
        twist.angular.z = 0.1
        self.cmd_pub.publish(twist)

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

