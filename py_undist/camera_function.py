import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from rosbags.image import message_to_cvimage
from cv_bridge import CvBridge
import numpy as np
import matplotlib.pyplot as plt
import time
from py_undist.file_handling import camera_name_calib_yaml_to_K_D

class CameraNode(Node):
    publishers = None
    def __init__(self):
        super().__init__('camera_node')
        # Define list of topics
        self.topic_names = ['/rgb_cam_fp_p/image_raw',
                        '/rgb_cam_fp_f/image_raw',
                        '/rgb_cam_fs_f/image_raw',
                        '/rgb_cam_fs_s/image_raw',
                        '/rgb_cam_ap_p/image_raw', 
                        '/rgb_cam_ap_a/image_raw',
                        '/rgb_cam_as_a/image_raw',
                        '/rgb_cam_as_s/image_raw']

        self.camera_K_and_D= {}
        
        self.publishers = {}

        for topic_name in self.topic_names:
            camera_name = topic_name[9:13]
            print(camera_name)
            K,D = camera_name_calib_yaml_to_K_D(camera_name)
            self.camera_K_and_D[camera_name] = {'K':K,'D':D}

            pub_name = 'undistorted_im_' + camera_name   
            pub = self.create_publisher(Image, pub_name,
                                        QoSProfile(depth=10,
                                        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                                        reliability=QoSReliabilityPolicy.BEST_EFFORT))
            self.publishers[camera_name] = pub
        self.filepath = "/home/mathias/Documents/Master_Thesis/Rosbags/rosbag2_2022_10_06-13_50_07_0"
        # self.rosbag_reader = Reader(self.filepath)
        self.bridge = CvBridge()

    def undistort(self, cv_image, camera_name):
        undistorted_image = None
        try:
            # Undistort image using camera calibration parameters
            K = self.camera_K_and_D[camera_name]['K']
            D = self.camera_K_and_D[camera_name]['D']
            undistorted_image = cv2.undistort(cv_image, K, D)
        except Exception as e:
            self.get_logger().error('Error undistorting image: %s' % str(e))
        return undistorted_image
    
    def msg_to_cv(self, msg):
        cv_image = None
        try:
            # Convert ROS message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8',)
        except Exception as e:
            self.get_logger().error('Error converting ROS message to OpenCV image: %s' % str(e))
        return cv_image
    
    def cv_to_msg(self, cv_image):
        msg = None
        try:
            # Convert OpenCV image to ROS message
            msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='rgb8')
        except Exception as e:
            self.get_logger().error('Error converting OpenCV image to ROS message: %s' % str(e))
        return msg
    
    def run(self):
        start_time = time.time()
        msg_count = 0
        undistorted_messages = list()
        camera_count = [None]*8
        # rate = self.create_rate(5)
        while True:
            with Reader(self.filepath) as reader:

                connections = [x for x in reader.connections if x.topic in self.topic_names]

                for connection, timestamp, rawdata in reader.messages(connections=connections):
                    msg = deserialize_cdr(rawdata, connection.msgtype)
                    if msg is None:
                        return
                    
                    camera_name = msg.header.frame_id[8:]
                    # Image and msg handeling
                    cv_image = self.msg_to_cv(msg)
                    undistorted_image = self.undistort(cv_image, camera_name)
                    undistorted_msg = self.cv_to_msg(undistorted_image)

                    undistorted_msg.header.frame_id = camera_name
                    # print('header ' + undistorted_msg.header.frame_id)

                    undistorted_messages.append(undistorted_msg)
                    msg_count += 1

                    if msg_count == 8:
                        # print('8 images read')
                        for msg in undistorted_messages:
                            camera_name = msg.header.frame_id
                            self.publishers[camera_name].publish(msg)
                        

                        # Check if we have reached the target 
                        # rate.sleep()
                        elapsed_time = time.time() - start_time
                        target_fps = 5
                        target_time_per_image = 1 / target_fps
                        if elapsed_time < target_time_per_image:
                            time.sleep(target_time_per_image - elapsed_time)
                            print('Slept for ' + str(target_time_per_image - elapsed_time))
                        # Reset the counter and start time
                        msg_count = 0
                        undistorted_messages = []
                        start_time = time.time()

def main(args=None):
    rclpy.init(args=args)
    camera_node = CameraNode()
    try:
        camera_node.run()
    except KeyboardInterrupt:
        pass
    finally:
        camera_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
