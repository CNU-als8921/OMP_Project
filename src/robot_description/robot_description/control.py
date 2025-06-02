import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import cv2
import numpy as np

class RedObjectAvoider(Node):
    def __init__(self):
        super().__init__('red_object_avoider')
        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/world/my_world/model/box_usv/link/camera_link/sensor/camera_sensor/image',
            self.image_callback,
            10)
        
        self.pub_thruster_L = self.create_publisher(Float64, '/box_usv/thruster_L', 10)
        self.pub_thruster_R = self.create_publisher(Float64, '/box_usv/thruster_R', 10)

    def image_callback(self, msg):
        # 이미지 변환
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w, _ = cv_image.shape
        center_x = w // 2

        # 빨간색 HSV 범위 정의
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            (x, y, rw, rh) = cv2.boundingRect(largest)
            obj_center = x + rw // 2

            left_margin = int(w * 0.2)
            right_margin = int(w * 0.8)

            if obj_center <= left_margin or obj_center >= right_margin:
                self.get_logger().info('Red object at edge → Go forward')
                self.go_forward()
            elif obj_center > center_x:
                self.get_logger().info('Red object on right → Turn left')
                self.avoid_left()
            else:
                self.get_logger().info('Red object on left → Turn right')
                self.avoid_right()
        else:
            self.go_forward()

    def avoid_right(self):
        self.pub_thruster_L.publish(Float64(data=0.0))
        self.pub_thruster_R.publish(Float64(data=5.0))

    def avoid_left(self):
        self.pub_thruster_L.publish(Float64(data=5.0))
        self.pub_thruster_R.publish(Float64(data=0.0))

    def go_forward(self):
        self.pub_thruster_L.publish(Float64(data=5.0))
        self.pub_thruster_R.publish(Float64(data=5.0))

def main(args=None):
    rclpy.init(args=args)
    node = RedObjectAvoider()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()