#!/usr/bin/env python3
"""
blue_can_detector_node.py

ROS2 node for blue soda can detection via HSV segmentation.
Subscribes to a camera topic (default /camera/image_raw), publishes std_msgs/Bool on /can_detected,
publishes Twist to /cmd_vel when a large can is detected, republishes annotated Image on /debug/can_image,
and displays the annotated image in an OpenCV window.
"""
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class BlueCanDetectorNode(Node):
    def __init__(self):
        super().__init__('obstacle_detect_node')

        # parameters
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('min_area', 500)
        self.declare_parameter('lower_blue_h', 100)
        self.declare_parameter('lower_blue_s', 150)
        self.declare_parameter('lower_blue_v', 50)
        self.declare_parameter('upper_blue_h', 140)
        self.declare_parameter('upper_blue_s', 255)
        self.declare_parameter('upper_blue_v', 255)
        self.declare_parameter('morph_open_iter', 2)
        self.declare_parameter('morph_dilate_iter', 1)
        self.declare_parameter('big_area_thresh', 10000)
        self.declare_parameter('drive_speed', 0.2)
        self.declare_parameter('drive_duration', 2.0)
        self._load_params()

        # OpenCV window for debug display
        cv2.namedWindow('Debug Can Image', cv2.WINDOW_NORMAL)

        self.bridge = CvBridge()
        self.prev_gray = None
        self.frame_idx = 0
        self.moving_forward = False
        self.drive_timer = None

        # ROS interfaces
        self.sub = self.create_subscription(
            Image, self.camera_topic, self.image_callback, 1)
        self.pub_flag = self.create_publisher(Bool, '/can_detected', 1)
        self.pub_img  = self.create_publisher(Image, '/debug/can_image', 1)
        self.pub_vel  = self.create_publisher(Twist, '/cmd_vel', 1)

        self.add_on_set_parameters_callback(self._on_params_update)
        self.get_logger().info(f'BlueCanDetectorNode started, subscribing to {self.camera_topic}')

    def _load_params(self):
        p = self.get_parameter
        self.camera_topic = p('camera_topic').get_parameter_value().string_value
        self.min_area = p('min_area').get_parameter_value().integer_value
        lh = p('lower_blue_h').get_parameter_value().integer_value
        ls = p('lower_blue_s').get_parameter_value().integer_value
        lv = p('lower_blue_v').get_parameter_value().integer_value
        uh = p('upper_blue_h').get_parameter_value().integer_value
        us = p('upper_blue_s').get_parameter_value().integer_value
        uv = p('upper_blue_v').get_parameter_value().integer_value
        self.lower_blue = np.array([lh, ls, lv])
        self.upper_blue = np.array([uh, us, uv])
        self.open_iter = p('morph_open_iter').get_parameter_value().integer_value
        self.dilate_iter = p('morph_dilate_iter').get_parameter_value().integer_value
        self.big_thresh = p('big_area_thresh').get_parameter_value().integer_value
        self.drive_speed = p('drive_speed').get_parameter_value().double_value
        self.drive_duration = p('drive_duration').get_parameter_value().double_value

    def _on_params_update(self, params):
        self._load_params()
        return SetParametersResult(successful=True)

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)

        # denoise mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=self.open_iter)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=self.dilate_iter)

        # find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected = False
        big_detected = False
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area >= self.min_area:
                x, y, w, h = cv2.boundingRect(largest)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                detected = True
                if area >= self.big_thresh:
                    big_detected = True

        # publish detection flag
        self.pub_flag.publish(Bool(data=detected))

        # drive if big can detected
        if big_detected and not self.moving_forward:
            self._start_drive()

        # annotate status
        status = "CAN DETECTED" if detected else "CLEAR"
        color = (255, 0, 0) if detected else (0, 255, 0)
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # republish debug image
        self.pub_img.publish(self.bridge.cv2_to_imgmsg(frame, 'bgr8'))

        # display annotated frame
        cv2.imshow('Debug Can Image', frame)
        cv2.waitKey(1)

    def _start_drive(self):
        twist = Twist()
        twist.linear.x = self.drive_speed
        self.pub_vel.publish(twist)
        self.moving_forward = True
        self.drive_timer = self.create_timer(self.drive_duration, self._stop_drive)

    def _stop_drive(self):
        twist = Twist()
        twist.linear.x = 0.0
        self.pub_vel.publish(twist)
        self.moving_forward = False
        if self.drive_timer:
            self.drive_timer.cancel()
            self.drive_timer = None

def main(args=None):
    rclpy.init(args=args)
    node = BlueCanDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutdown by user')
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
