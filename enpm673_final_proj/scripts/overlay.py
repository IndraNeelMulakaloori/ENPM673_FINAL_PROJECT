#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int64MultiArray
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
import cv2

class BoxOverlay(Node):
    def __init__(self):
        super().__init__('box_overlay')
        self.bridge = CvBridge()
        self.box = None
        self.create_subscription(Int64MultiArray, '/box_stop', self.box_cb, 10)
        self.create_subscription(Image, '/camera/image_raw', self.img_cb, 10)

        self.create_subscription(Int64MultiArray,'/box_stop',self.box_cb,qos_profile_sensor_data)

    def box_cb(self, msg):
         self.box = msg.data
         self.get_logger().info(f"Got box: {self.box}")

    def img_cb(self, msg):
        self.get_logger().info("Overlay got image frame")
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        if self.box:
            x1,y1,x2,y2 = self.box
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
            cv2.putText(img, 'STOP', (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        cv2.imshow('Overlay', img)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = BoxOverlay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()
