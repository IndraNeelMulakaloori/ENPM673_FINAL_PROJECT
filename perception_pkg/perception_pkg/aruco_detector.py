#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from tf2_ros import TransformBroadcaster, TransformListener, Buffer
from geometry_msgs.msg import TransformStamped
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy, HistoryPolicy

from rclpy.logging import get_logger

class ArucoDetectorNode(Node):
    def __init__(self,node_name):
        super().__init__(node_name=node_name)
        self.get_logger().info("Aruco Detector Node Initialized")
        
         # Declare parameters with default values
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("marker_length", 0.1)
        self.declare_parameter("aruco_dictionary_id", "DICT_4X4_50")
        self.declare_parameter("debug_view", True)

        # Get parameters
        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.marker_length = self.get_parameter("marker_length").get_parameter_value().double_value
        self.aruco_dict = self.get_parameter("aruco_dictionary_id").get_parameter_value().string_value
        self.debug_view = self.get_parameter("debug_view").get_parameter_value().bool_value

        self._image_bridge = CvBridge()
        
        self._img_topic_sub = self.create_subscription(msg_type = Image,
         topic = self.image_topic,
         callback=self.image_callback,
         qos_profile=10)
        
        
        self.get_logger().info(f"Subscribed to image: {self.image_topic}, Marker Length: {self.marker_length}, ArUco Dictionary: {self.aruco_dict}, Debug view: {self.debug_view}")
        
       
    def image_callback(self,msg):
            try:
                cv_image = self._image_bridge.imgmsg_to_cv2(msg, "bgr8")
            except Exception as e:
                self.get_logger().error(f"Error converting image: {e}")
                return

            aruco_dict_id = getattr(cv2.aruco, self.aruco_dict)
            aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_id)
            aruco_params = cv2.aruco.DetectorParameters_create()
            
            # Detect markers
            corners, ids, _ = cv2.aruco.detectMarkers(cv_image, aruco_dict, parameters=aruco_params)
            
            if ids is not None:
                self.get_logger().info(f"[Aruco] Marker(s) detected! IDs: {ids.flatten()}")
            else:
                self.get_logger().info("[Aruco] No marker detected.")
            
            

def main(args=None):
    
    get_logger('hello').info("Aruco Detector Node Starting")
    try:
        # Initialize ROS2 communication
        rclpy.init(args=args)
        node = ArucoDetectorNode('aruco_detector')
        # Create and initialize the node
        # Spin the node to process callbacks
        
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Handle clean Ctrl+C exit gracefully
        print("Node stopped cleanly by user")
    except Exception as e:
        # Catch and report any other exceptions
        print(f"Error occurred: {e}")
        node.get_logger().error(f"Error occurred: {e}")
    finally:
        # Cleanup resources properly
        if "node" in locals() and rclpy.ok():
            node.destroy_node()
        # Only call shutdown if ROS is still initialized
        if rclpy.ok():
            rclpy.shutdown()