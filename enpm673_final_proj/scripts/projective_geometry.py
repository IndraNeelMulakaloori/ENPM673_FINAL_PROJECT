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

class ProjectiveGeometryNode(Node):
    def __init__(self,node_name):
        super().__init__(node_name=node_name)
        # self.get_logger().info("Projective Geometry Node Initialized")
        
        # Declare parameters with default values
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        
        # Get parameters
        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        
        
        self._image_bridge = CvBridge()
        
        ## Create subscriptions to /image_raw and /camera_info
        self._img_topic_sub = self.create_subscription(msg_type = Image,
         topic = self.image_topic,
         callback=self.image_callback,
         qos_profile=10)
        
        
        
        
        # self.get_logger().info(f"Subscribed to image: {self.image_topic}, Camera Info: {self.camera_info_topic}")
    
    # Checkerboard 6 x 8, cell/block size = 0.03m x 0.03m
    def compute_intersection(self, line1, line2):
            vx1, vy1, x1, y1 = line1.flatten()
            vx2, vy2, x2, y2 = line2.flatten()
            A = np.array([[vx1, -vx2], [vy1, -vy2]])
            b = np.array([x2 - x1, y2 - y1])
            if np.linalg.det(A) != 0:
                t = np.linalg.solve(A, b)
                px = x1 + vx1 * t[0]
                py = y1 + vy1 * t[0]
                return int(px), int(py)
            return None
    def image_callback(self,msg):
            try:
                cv_image = self._image_bridge.imgmsg_to_cv2(msg, "bgr8")
            except Exception as e:
                # self.get_logger().error(f"Error converting image: {e}")
                return

            # Process the image here
            # For example, you can display it using OpenCV
            # Checkerboard 6 x 8, cell/block size = 0.03m x 0.03m
            ## 198 366 265 78
            square_size = 30
            checkerboard_size = (7,5)
            # img_copy = np.copy(cv_image)
            # roi = cv2.selectROI("Select ROI", cv_image, fromCenter=False)
            # roi = (198,363,265,78)
            # cropped = cv_image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
            # self.get_logger().info(f"{roi}")
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.GaussianBlur(gray_image, (7,7),sigmaX=0)
            # edges = cv2.Canny(gray_image, 100, 150)
            # Lists to store object points and image points
            objpoints = []  # 3D points in real world space
            imgpoints = []  # 2D points in image plane
            corner_detected_indices = []
            # linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)
            # self.get_logger().info(f"Hough lines detected : {linesP}")
            # cv2.namedWindow('Canny Edge Detection')
            # def nothing(x):
            #         pass
            # # Create trackbars for thresholds
            # cv2.createTrackbar('Lower', 'Canny Edge Detection', 50, 500, nothing)
            # cv2.createTrackbar('Upper', 'Canny Edge Detection', 150, 600, nothing)

            # while True:
            #     lower = cv2.getTrackbarPos('Lower', 'Canny Edge Detection')
            #     upper = cv2.getTrackbarPos('Upper', 'Canny Edge Detection')

            #     edges = cv2.Canny(img_copy, lower, upper)

            #     cv2.imshow('Canny Edge Detection', edges)

            #     if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            #         break
            
            # if linesP is not None:
            #     for i in range(0, len(linesP)):
            #         x1,y1,x2,y2 = linesP[i][0]
            # #         slope = (y2-y1)/(x2-x1)
            # #         intercept = y1 - slope * x1
            #         cv2.line(img_copy, (x1, y1), (x2, y2), (0,0,255), 2, cv2.LINE_AA)

            # cv2.imshow("Image", gray_blur)
            # cv2.imshow("New_Image",edges)
            
            # Find chessboard corners using cv.findChessboardCorners
            ret, corners = cv2.findChessboardCorners(gray_image, checkerboard_size,cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
            # If corners are found
            # if ret :
            #     self.get_logger().info(f"No of corners found : {len(corners)}")
            if ret:
                # Refine corner locations
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                refined_corners = cv2.cornerSubPix(gray_image, corners, (9, 9), (-1, -1), criteria)
                # Append object points and image points
                # objpoints.append(objp)
                # ## Refined corners need to reshape into (-1,2) for further computation
                # imgpoints.append(refined_corners.reshape(-1,2))
                # Draw th corners and append the foudn corner indices
                # Row-wise lines
                # cv2.drawChessboardCorners(cv_image, checkerboard_size, refined_corners, ret)
                reshaped = refined_corners.reshape(checkerboard_size[1], checkerboard_size[0], 2)  # (rows, cols, 2)
                for row in reshaped:
                    [vx, vy, x0, y0] = cv2.fitLine(row, cv2.DIST_L2, 0, 0.01, 0.01)
                    x1, y1 = int(x0 - vx * 1000), int(y0 - vy * 1000)
                    x2, y2 = int(x0 + vx * 1000), int(y0 + vy * 1000)
                    cv2.line(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 1)

                # Column-wise lines
                for col in reshaped.transpose((1, 0, 2)):
                    [vx, vy, x0, y0] = cv2.fitLine(col, cv2.DIST_L2, 0, 0.01, 0.01)
                    x1, y1 = int(x0 - vx * 1000), int(y0 - vy * 1000)
                    x2, y2 = int(x0 + vx * 1000), int(y0 + vy * 1000)
                    cv2.line(cv_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
            # cv2.imshow("Gray Blurred", gray_image)
            # Compute vanishing points
                row_vp = self.compute_intersection(
                    cv2.fitLine(reshaped[0], cv2.DIST_L2, 0, 0.01, 0.01),
                    cv2.fitLine(reshaped[-1], cv2.DIST_L2, 0, 0.01, 0.01)
                )

                col_vp = self.compute_intersection(
                cv2.fitLine(reshaped[:, 0, :], cv2.DIST_L2, 0, 0.01, 0.01),
                cv2.fitLine(reshaped[:, -1, :], cv2.DIST_L2, 0, 0.01, 0.01)
            )

                if row_vp and col_vp:
                    cv2.line(cv_image, row_vp, col_vp, (0, 255, 255), 2)
                    # self.get_logger().info(f"Horizon line drawn between {row_vp} and {col_vp}")
            cv2.imshow("Hough Lines",cv_image)
            cv2.waitKey(1)
            # self.get_logger().info("Image received and processed")
            
            
            

def main(args=None):
    

    try:
        # Initialize ROS2 communication
        rclpy.init(args=args)
        node = ProjectiveGeometryNode('projective_geometry')
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

if __name__ == '__main__':
    main()