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

class UndistortNode(Node):
    subscriptions = None
    publishers = None

    def __init__(self):
        super().__init__('undistort_node')
        # Define list of topics
        topic_names = ['/rgb_cam_fp_p/image_raw',
                        '/rgb_cam_fp_f/image_raw',
                        '/rgb_cam_fs_f/image_raw',
                        '/rgb_cam_fs_s/image_raw',
                        '/rgb_cam_ap_p/image_raw', 
                        '/rgb_cam_ap_a/image_raw',
                        '/rgb_cam_as_a/image_raw',
                        '/rgb_cam_as_s/image_raw']

        self.camera_K_and_D= {}
        self.subscriptions = []
        self.publishers = {}

        for topic_name in topic_names:
            camera_name = topic_name[9:13]
            print(camera_name)
            K,D = camera_name_calib_yaml_to_K_D(camera_name)
            self.camera_K_and_D[camera_name] = {'K':K,'D':D}
            
            sub = self.create_subscription(Image, topic_name,self.callback,
                                           QoSProfile(depth=10,
                                            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                                            reliability=QoSReliabilityPolicy.BEST_EFFORT))
            self.subscriptions.append(sub)

            pub_name = 'undistorted_im_' + camera_name   
            pub = self.create_publisher(Image, pub_name,
                                        QoSProfile(depth=10,
                                        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                                        reliability=QoSReliabilityPolicy.BEST_EFFORT))
            self.publishers[camera_name] = pub
        # print(self.publishers)
        
        # self.subscription = self.create_subscription(
        #     Image,
        #     '/rgb_cam_fp_p/image_raw',
        #     self.callback,
        #     QoSProfile(depth=10,
        #                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        #                reliability=QoSReliabilityPolicy.RELIABLE))

        # self.publisher = self.create_publisher(
        #     Image,
        #     'undist_im',
        #     QoSProfile(depth=10,
        #                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        #                reliability=QoSReliabilityPolicy.BEST_EFFORT))

        
      
        self.bridge = CvBridge()

    def callback(self, msg):
        camera_name = msg.header.frame_id[8:]
        # Convert ROS message to OpenCV image
        cv_image = self.msg_to_cv(msg)
        # Undistort image
        undistorted_image = self.undistort(cv_image, camera_name)
        # Convert OpenCV image to ROS message
        undistorted_msg = self.cv_to_msg(undistorted_image)
        # Add header
       
        undistorted_msg.header.frame_id = camera_name
        
        undistorted_msg.header.stamp = self.get_clock().now().to_msg()
        # Publish undistorted image
        self.publishers[camera_name].publish(undistorted_msg)

        

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

def main(args=None):
    rclpy.init(args=args)

    node = UndistortNode()
    file_path = "/home/mathias/Documents/Master_Thesis/Rosbags/rosbag2_2022_10_06-13_50_07_0"
    # topic_names = ['/rgb_cam_fp_p/image_raw']
    topic_names = ['/rgb_cam_fp_p/image_raw',
                    '/rgb_cam_fp_f/image_raw',
                    '/rgb_cam_fs_f/image_raw',
                    '/rgb_cam_fs_s/image_raw',
                    '/rgb_cam_ap_p/image_raw', 
                    '/rgb_cam_ap_a/image_raw',
                    '/rgb_cam_as_a/image_raw',
                    '/rgb_cam_as_s/image_raw']
    # Open rosbag file
    # bag = Reader(file_path)

    # # Read messages from rosbag and publish them
    # for topic, msg, t in bag.read_messages():
    #     if topic == topic_name:
    #         msg
    #         node.callback(msg)
     
    # Spin node until shutdown
    skip_im = 0

    with Reader(file_path) as reader:

        connections = [x for x in reader.connections if x.topic in topic_names]

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = deserialize_cdr(rawdata, connection.msgtype)
            # im = message_to_cvimage(msg,'rgb8')
            time.sleep(0.250)
            
            if msg.header.frame_id == 'rgb_cam_fp_p':
                print('fp_p')
                skip_im = skip_im +1
                if skip_im % 5 == 0:
                    print('Skipped im')
                    continue

            # plt.figure()
            # plt.imshow(im)
            
            # plt.figure()
            # plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            # plt.show()
            node.callback(msg)
    try:
        rclpy.spin(node) # sjekk ut spin_once og at den kjører noden inni loopen ikke utafor . Se på rospy rate
    except KeyboardInterrupt:
        pass
    # counter = 0
    # rate = node.create_rate(5)
    # try:
    #     with Reader(file_path) as reader:

    #         while rclpy.ok():
    #             # Read 8 messages from the file
    #             connections = [x for x in reader.connections if x.topic in topic_names]
    #             messages = []
    #             for connection, timestamp, rawdata in reader.messages(connections=connections):
    #                 messages.append(deserialize_cdr(rawdata, connection.msgtype))
                    
    #                 counter += 1
    #                 if counter == 8:
    #                     for msg in messages:
    #                         node.callback(msg)
    #                         rclpy.spin_once(node)
    #                     messages = []
                    
                        
                
    # except KeyboardInterrupt:
    #     pass
            

    # Shutdown the node after all messages have been processed
    node.destroy_node()
    rclpy.shutdown()
    



if __name__ == '__main__':
    main()
