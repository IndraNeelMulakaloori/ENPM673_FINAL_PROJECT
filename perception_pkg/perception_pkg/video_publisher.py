import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class VideoPublisher(Node):
    def __init__(self):
        super().__init__('video_publisher')
        self.publisher_ = self.create_publisher(Image, 'video_frames', qos_profile=10)
        self.timer = self.create_timer(1/2, self.timer_callback)  # 2 Hz
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture('src/ENPM673_turtlebot_perception_challenge/videos/test_run.mp4')

        if not self.cap.isOpened():
            self.get_logger().error('Failed to open video file.')

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.publisher_.publish(msg)
            self.get_logger().info('Published frame')
        else:
            self.get_logger().info('Video ended or frame not read.')
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop the video

def main(args=None):
    rclpy.init(args=args)
    node = VideoPublisher()
    rclpy.spin(node)
    node.cap.release()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
