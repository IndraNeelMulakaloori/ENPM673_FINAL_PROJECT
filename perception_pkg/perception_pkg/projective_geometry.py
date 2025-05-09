#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

from rclpy.logging import get_logger

class ProjectiveGeometryNode(Node):
    def __init__(self,node_name):
        super().__init__(node_name=node_name)
        self.get_logger().info("Projective Geometry Node Initialized")
        
        # Declare parameters with default values
        self.declare_parameter("image_topic", "video_frames")
        # self.declare_parameter("image_topic", "/camera/image_raw")
        # self.declare_parameter("image_topic", "/camera/image_raw/compressed")   
        # self.declare_parameter("camera_info_topic", "/camera/camera_info")                                   
        # self.declare_parameter("image_topic","/tb4_2/oakd/rgb/image_raw")
        # self.declare_parameter("camera_info_topic","/tb4_2/oakd/rgb/camera_info")
        
        
        # Get parameters
        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        # self.camera_info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        
        
        self._image_bridge = CvBridge()
        
        ## Create subscriptions to /image_raw and /camera_info
        self._img_topic_sub = self.create_subscription(msg_type = Image,
         topic = self.image_topic,
         callback=self.image_callback,
         qos_profile=10)
        
        
        
        
        # self.get_logger().info(f"Subscribed to image: {self.image_topic}, Camera Info: {self.camera_info_topic}")
        self.get_logger().info(f"Subscribed to image: {self.image_topic}")
    
    
    def compute_intersection(self, line1, line2):
        """
        For every first and last lne of row and col
        this function computes the intersection of the both 
        lines using directional vectors and points (parametric forms).
        
        This computes using linear algebra matrices
        Ax=b

        Args:
            line1 (cv.fitline): array lines that consist (dx,dy,x,y)
            line2 (cv.fitline): array lines that consist (dx,dy,x,y)

        Returns:
            int : points 
        """
        dx1, dy1, x1, y1 = line1.flatten()  ## flatten lines and returns directional vectors and points
        dx2, dy2, x2, y2 = line2.flatten()  ## flatten lines and returns directional vectors and points
        A = np.array([[dx1, -dx2], [dy1, -dy2]])     
        b = np.array([x2 - x1, y2 - y1])     
        if np.linalg.det(A) != 0:
            t = np.linalg.solve(A, b)
            px = x1 + dx1 * t[0]
            py = y1 + dy1 * t[0]
            return int(px), int(py)
        return None
    def image_callback(self,msg):
            try:
                cv_image = self._image_bridge.imgmsg_to_cv2(msg, "bgr8")
            except Exception as e:
                self.get_logger().error(f"Error converting image: {e}")
                return

            # Process the image here
            # For example, you can display it using OpenCV
            # Checkerboard 6 x 8, cell/block size = 0.03m x 0.03m
            ## 198 366 265 78
            square_size = 30
            # checkerboard_size = (7,5)
            checkerboard_size = (7,4) ## Video thingy
           
            border_size = 500
            cv_image = cv2.copyMakeBorder(cv_image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.GaussianBlur(gray_image, (7,7),sigmaX=0)
            
            # Find chessboard corners using cv.findChessboardCorners
            ret, corners = cv2.findChessboardCorners(gray_image, checkerboard_size,cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
            # If corners are found
            if ret :
                self.get_logger().info(f"No of corners found : {len(corners)}")
            if ret:
                # Refine corner locations
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                refined_corners = cv2.cornerSubPix(gray_image, corners, (9, 9), (-1, -1), criteria)
            
                # Row-wise lines
                # cv2.drawChessboardCorners(cv_image, checkerboard_size, refined_corners, ret)
                reshaped = refined_corners.reshape(checkerboard_size[1], checkerboard_size[0], 2)  # (rows, cols, 2)
                scale = 1000
                
                ### For every row , column we are fitting the lines 
                ### Using cv2.fitline - least squares method
                ### This returns us a mean or centroid of the data points(checkerboard coords)
                ### and directional vector(dx,dy)
                ### Using this data we will be scaling the lines in their respective directions for every row./col
                for row in reshaped:
                    [dx, dy, x0, y0] = cv2.fitLine(row, cv2.DIST_L2, 0, 0.01, 0.01)
                    x1, y1 = int(x0 - dx * (scale)), int(y0 - dy * (scale))
                    x2, y2 = int(x0 + dx * (scale)), int(y0 + dy * (scale))
                    cv2.line(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 1)

                # Column-wise lines
                for col in reshaped.transpose((1, 0, 2)):
                    [dx, dy, x0, y0] = cv2.fitLine(col, cv2.DIST_L2, 0, 0.01, 0.01)
                    x1, y1 = int(x0 - dx * scale), int(y0 - dy * scale)
                    x2, y2 = int(x0 + dx * scale), int(y0 + dy * scale)
                    cv2.line(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 1)
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
                    # dx = col_vp[0] - row_vp[0]
                    # dy = col_vp[1] - row_vp[1]
                    # x_row,y_row = row_vp
                    
                    # pt1 = (int(x_row - dx), int(y_row - dy ))
                    # pt2 = (int(x_row + dx), int(y_row + dy ))
                    # cv2.line(cv_image, pt1, pt2, (0, 255, 255), 2)
                    
                    
                    ### Computing midpoint of both vanishing points
                    ### and extrending them in bot h directions
                    ### this is for having control of visual representation
                    ### alternatively u can use cv line from row_vp to col_vp
                    # mid_x = (row_vp[0] + col_vp[0]) // 2
                    # mid_y = (row_vp[1] + col_vp[1]) // 2
                    # dx = col_vp[0] - row_vp[0]
                    # dy = col_vp[1] - row_vp[1]
                
                    # shrink = 0.5  # This decides the line scaling
                    # pt1 = (int(mid_x - dx * shrink), int(mid_y - dy * shrink))
                    # pt2 = (int(mid_x + dx * shrink), int(mid_y + dy * shrink))
                    
                    # cv2.circle(cv_image,pt1,radius=10,color=(255,255,255),thickness=-1)
                    # cv2.circle(cv_image,pt2,radius=10,color=(255,255,255),thickness=-1)
                    # cv2.line(cv_image, pt1, pt2, (0, 255, 255), 2)
                    cv2.circle(cv_image,row_vp,radius=10,color=(255,255,255),thickness=-1)
                    cv2.circle(cv_image,col_vp,radius=10,color=(255,255,255),thickness=-1)
                    cv2.line(cv_image, row_vp, col_vp, (0, 255, 255), 2)
                    self.get_logger().info(f"Horizon line drawn between {row_vp} and {col_vp}")
            ## Screen Adjustment
            screen_res = 1280, 720  # or use pyautogui.size() if needed
            scale_width = screen_res[0] / cv_image.shape[1]
            scale_height = screen_res[1] / cv_image.shape[0]
            scale = min(scale_width, scale_height)
            display_image = cv2.resize(cv_image, None, fx=scale, fy=scale)
            cv2.imshow("Projective Geometry", display_image)
            cv2.waitKey(1)
            self.get_logger().info("Image received and processed")
            
            
            

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