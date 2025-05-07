# #!/usr/bin/env python3
# import sys

# from enpm673_module.enpm673_final_proj import *

# def main():
#     print('Hi from enpm673_final_proj script.')
#     printHello()


# if __name__ == '__main__':
#     main()



#!/usr/bin/env python3
# filepath: /home/shreya/ros2_ws/src/ENPM673_FINAL_PROJECT/enpm673_final_proj/scripts/turtlebot_controller.py

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from rclpy.qos import qos_profile_sensor_data

class TurtlebotController(Node):
    def __init__(self):
        super().__init__('turtlebot_controller')
        # Publisher for velocity commands
        self.vel_publisher = self.create_publisher(
            Twist, 
            '/cmd_vel', 
            10
        )
        
        # Subscriber for stop signals
        self.stop_subscriber = self.create_subscription(
            Bool,
            '/stop',
            self.stop_callback,
            qos_profile_sensor_data
        )
        
        # Timer for publishing velocity commands
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # State variable for stop signal
        self.should_stop = False
        self.get_logger().info('Turtlebot controller started')

    def stop_callback(self, msg):
        # Update stop state based on received message
        self.should_stop = msg.data
        if self.should_stop:
            self.get_logger().info('Received STOP signal')
            # Immediately send a stop command
            self.send_stop_command()
        else:
            self.get_logger().info('Received GO signal')

    def timer_callback(self):
        # Create Twist message
        twist_msg = Twist()
        
        if not self.should_stop:
            # Move forward if not stopped
            twist_msg.linear.x = 0.2  # 0.2 m/s forward
            twist_msg.angular.z = 0.0  # No rotation
        else:
            # Stop the robot
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0
            
        # Publish velocity command
        self.vel_publisher.publish(twist_msg)

    def send_stop_command(self):
        # Immediately send a zero velocity command to stop
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = 0.0
        self.vel_publisher.publish(twist_msg)

def main(args=None):
    rclpy.init(args=args)
    controller = TurtlebotController()
    rclpy.spin(controller)
    
    # Cleanup
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()