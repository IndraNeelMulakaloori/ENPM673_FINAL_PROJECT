import cv2 
from matplotlib import pyplot as plt 
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from ultralytics import YOLO
from std_msgs.msg import Bool
from std_msgs.msg import Int64MultiArray
from sensor_msgs.msg import Image

class YOLOV8(Node):

    def __init__(self):
        super().__init__('yolo_v8')
        ## Give the path to the model file. Please replace this with the path to the model file. 
        yolo_model_path = '/home/shreya/ros2_ws/src/ENPM673_FINAL_PROJECT/enpm673_final_proj/models/yolov8s.pt'
        ## Yolo model is initialized by loading the model specified in the path
        self.model = YOLO(yolo_model_path)

        print("Classes:", self.model.names)
        
        ## A threshold for obstacle detection is declared        
        self.threshold = 0.7
        
        ## Subscriber for turtlebot camera topic to get the image        
        self.subscription = self.create_subscription(Image,'/camera/image_raw',self.camera_callback,qos_profile_sensor_data)

        ## Publisher to publish the flag value for stopping the robot when a stop sign is detected               
        self.publisher_stop = self.create_publisher(Bool,'/stop',qos_profile_sensor_data)
        ## Publisher to publish the bounding box values                
        self.publisher_box = self.create_publisher(Int64MultiArray,'/box_stop',qos_profile_sensor_data)
        ## Initializing a CV bridge to convert sensor_msgs Image to a opencv readable image format              
        self._bridge = CvBridge()


    def camera_callback(self, msg):
        stop_msg = Bool()
        box_msg  = Int64MultiArray()

        # Convert ROS Image → OpenCV BGR
        img = self._bridge.imgmsg_to_cv2(msg, "bgr8")

        # Run YOLO inference
        results = self.model(img)[0]

        # # If any detection above threshold, publish and return
        # for x1, y1, x2, y2, score, class_id in results.boxes.data.tolist():
        #     if score >= self.threshold:
        #         box_msg.data  = [int(x1), int(y1), int(x2), int(y2)]
        #         stop_msg.data = True
        #         self.publisher_box.publish(box_msg)
        #         self.publisher_stop.publish(stop_msg)
        #         return


        # COCO class index for stop‐sign is 12
        STOP_CLASS_ID = 11

        # Look for at least one stop‐sign
        for x1, y1, x2, y2, score, class_id in results.boxes.data.tolist():
            # 1) ignore everything except stop‐sign detections
            if int(class_id) != STOP_CLASS_ID:
                continue

            # 2) only consider hits above your confidence threshold
            if score >= self.threshold:
                box_msg.data  = [int(x1), int(y1), int(x2), int(y2)]
                stop_msg.data = True
                self.publisher_box.publish(box_msg)
                self.publisher_stop.publish(stop_msg)
                return

        # Otherwise clear the stop flag
        stop_msg.data = False
        self.publisher_stop.publish(stop_msg)
        # Clear previous bounding box by publishing empty data
        box_msg.data = []  # Empty array indicates no detection
        self.publisher_box.publish(box_msg)


## Main function of the script                           
def main(args=None):
    rclpy.init(args=args)
    
    ## Creating a Ros2 Node object
    yolo = YOLOV8()

    rclpy.spin(yolo)
    
    yolo.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
