import cProfile
import pstats
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from py_undist.calculate_bew_data import calculate_rgb_matrix_for_BEW, interpolate, make_final_mask_and_pixel_pos, interp_weights
from py_undist.config import BEW_IMAGE_HEIGHT, BEW_IMAGE_WIDTH
import time
from py_undist.cls_mA2 import mA2
from py_undist.cls_Camera import Camera
import numpy as np
from py_undist.file_handling import camera_name_calib_yaml_to_K_D
from cv_bridge import CvBridge
import select
import sys

def make_BEW(vessel_mA2: mA2):
    start = time.time()

    grid_x,grid_y = np.meshgrid(range(BEW_IMAGE_HEIGHT), range(BEW_IMAGE_WIDTH), indexing='ij')

    points = np.vstack((vessel_mA2.fp_p.pixel_positions_masked,
                        vessel_mA2.fp_f.pixel_positions_masked,
                        vessel_mA2.fs_f.pixel_positions_masked,
                        vessel_mA2.fs_s.pixel_positions_masked,
                        vessel_mA2.ap_p.pixel_positions_masked,
                        vessel_mA2.ap_a.pixel_positions_masked,
                        vessel_mA2.as_a.pixel_positions_masked,
                        vessel_mA2.as_s.pixel_positions_masked,
                        vessel_mA2.black_pixel_pos))

    rgb_fp_p = calculate_rgb_matrix_for_BEW(vessel_mA2.fp_p.im,vessel_mA2.fp_p.image_mask)
    rgb_fp_f = calculate_rgb_matrix_for_BEW(vessel_mA2.fp_f.im,vessel_mA2.fp_f.image_mask)
    rgb_fs_f = calculate_rgb_matrix_for_BEW(vessel_mA2.fs_f.im,vessel_mA2.fs_f.image_mask) 
    rgb_fs_s = calculate_rgb_matrix_for_BEW(vessel_mA2.fs_s.im,vessel_mA2.fs_s.image_mask) 
    rgb_ap_p = calculate_rgb_matrix_for_BEW(vessel_mA2.ap_p.im,vessel_mA2.ap_p.image_mask)
    rgb_ap_a = calculate_rgb_matrix_for_BEW(vessel_mA2.ap_a.im,vessel_mA2.ap_a.image_mask)
    rgb_as_a = calculate_rgb_matrix_for_BEW(vessel_mA2.as_a.im,vessel_mA2.as_a.image_mask)
    rgb_as_s = calculate_rgb_matrix_for_BEW(vessel_mA2.as_s.im,vessel_mA2.as_s.image_mask)

    rgb = np.vstack((rgb_fp_p,
                     rgb_fp_f,
                     rgb_fs_f, 
                     rgb_fs_s, 
                     rgb_ap_p, 
                     rgb_ap_a,
                     rgb_as_a,
                     rgb_as_s,
                     vessel_mA2.black_pixel_rgb))


    # Delaunay 4
    xy = points
    uv=np.zeros([grid_x.shape[0]*grid_y.shape[1],2])
    uv[:,0]=grid_y.ravel()
    uv[:,1]=grid_x.ravel()
    values = rgb
    
    if ((vessel_mA2.vtx is None) or (vessel_mA2.wts is None)):
        # Computed once and for all !
        print('Making verticies & weigts and saving them')
        vtx, wts = interp_weights(xy, uv)
        np.save('/home/mathias/Documents/ros2_ws_master/src/py_undist/py_undist/vtx.npy', vtx)
        np.save('/home/mathias/Documents/ros2_ws_master/src/py_undist/py_undist/wts.npy', wts)
        vessel_mA2.set_vtx(vtx)
        vessel_mA2.set_wts(wts)
        
    
    vtx = vessel_mA2.vtx
    wts = vessel_mA2.wts

    val_r = values[:,0].ravel()
    val_g = values[:,1].ravel()
    val_b = values[:,2].ravel()
    valuesi_r=interpolate(val_r, vtx, wts)
    valuesi_g=interpolate(val_g, vtx, wts)
    valuesi_b=interpolate(val_b, vtx, wts)


    valuesi_r = valuesi_r.reshape((grid_x.shape[0],grid_x.shape[1]),order='F')
    valuesi_r = valuesi_r.astype(np.uint8)
    
    valuesi_g= valuesi_g.reshape((grid_x.shape[0],grid_x.shape[1]),order='F')
    valuesi_g = valuesi_g.astype(np.uint8)
    
    valuesi_b= valuesi_b.reshape((grid_x.shape[0],grid_x.shape[1]),order='F')
    valuesi_b = valuesi_b.astype(np.uint8)
 
    im = np.dstack((valuesi_r,valuesi_g,valuesi_b))
    end = time.time()
    print('Time: ' + str((end-start)*1000) + ' ms')
    return im

def init_mA2():
    # print('Cores:'+str(multiprocessing.cpu_count())) = 12
    # print(scipy.__version__) 1.9.1
    # topic_names = ['/rgb_cam_fp_p/image_raw',
    #                '/rgb_cam_fs_f/image_raw',
    #                '/rgb_cam_fs_s/image_raw',
    #                '/rgb_cam_ap_p/image_raw', 
    #                '/rgb_cam_ap_a/image_raw',
    #                '/rgb_cam_as_a/image_raw',
    #                '/rgb_cam_as_s/image_raw']
                   # Missing '/rgb_cam_fp_p/image_raw' in rosbag2...80
    camera_fp_p = Camera('rgb_cam_fp_p')
    camera_fp_f = Camera('rgb_cam_fp_f')
    camera_fs_f = Camera('rgb_cam_fs_f')
    camera_fs_s = Camera('rgb_cam_fs_s')
    camera_ap_p = Camera('rgb_cam_ap_p')
    camera_ap_a = Camera('rgb_cam_ap_a')
    camera_as_f = Camera('rgb_cam_as_a')
    camera_as_s = Camera('rgb_cam_as_s')
    vessel_mA2 = mA2(camera_fp_p,camera_fp_f,camera_fs_f,camera_fs_s,camera_ap_p,camera_ap_a,camera_as_f,camera_as_s)

    return vessel_mA2

class BEWImage(Node):
    subscriptions = None

    def __init__(self):
        super().__init__('bew_image')
        topic_names = ['undistorted_im_fp_p',
                        'undistorted_im_fp_f',
                        'undistorted_im_fs_f',
                        'undistorted_im_fs_s',
                        'undistorted_im_ap_p', 
                        'undistorted_im_ap_a',
                        'undistorted_im_as_a',
                        'undistorted_im_as_s']
        self.vessel_mA2 = init_mA2()
        self.vessel_mA2.find_triangle_between_each_cameras()

        image_mask, pixel_pos_masked =make_final_mask_and_pixel_pos(self.vessel_mA2.fp_p.wall_dist_mask, self.vessel_mA2.fp_p.pixel_positions_I_BEW, self.vessel_mA2.fp_p.left_triangle, self.vessel_mA2.fp_p.right_triangle)
        self.vessel_mA2.fp_p.set_image_mask(image_mask)
        self.vessel_mA2.fp_p.set_pixel_positions_masked(pixel_pos_masked)

        image_mask, pixel_pos_masked =make_final_mask_and_pixel_pos(self.vessel_mA2.fp_f.wall_dist_mask, self.vessel_mA2.fp_f.pixel_positions_I_BEW, self.vessel_mA2.fp_f.left_triangle, self.vessel_mA2.fp_f.right_triangle)
        self.vessel_mA2.fp_f.set_image_mask(image_mask)
        self.vessel_mA2.fp_f.set_pixel_positions_masked(pixel_pos_masked)
        
        image_mask, pixel_pos_masked =make_final_mask_and_pixel_pos(self.vessel_mA2.fs_f.wall_dist_mask, self.vessel_mA2.fs_f.pixel_positions_I_BEW, self.vessel_mA2.fs_f.left_triangle, self.vessel_mA2.fs_f.right_triangle)
        self.vessel_mA2.fs_f.set_image_mask(image_mask)
        self.vessel_mA2.fs_f.set_pixel_positions_masked(pixel_pos_masked)
        
        image_mask, pixel_pos_masked =make_final_mask_and_pixel_pos(self.vessel_mA2.fs_s.wall_dist_mask, self.vessel_mA2.fs_s.pixel_positions_I_BEW, self.vessel_mA2.fs_s.left_triangle,self.vessel_mA2.fs_s.right_triangle)
        self.vessel_mA2.fs_s.set_image_mask(image_mask)
        self.vessel_mA2.fs_s.set_pixel_positions_masked(pixel_pos_masked)
        
        image_mask, pixel_pos_masked =make_final_mask_and_pixel_pos(self.vessel_mA2.ap_p.wall_dist_mask, self.vessel_mA2.ap_p.pixel_positions_I_BEW, self.vessel_mA2.ap_p.left_triangle, self.vessel_mA2.ap_p.right_triangle)
        self.vessel_mA2.ap_p.set_image_mask(image_mask)
        self.vessel_mA2.ap_p.set_pixel_positions_masked(pixel_pos_masked)
        
        image_mask, pixel_pos_masked =make_final_mask_and_pixel_pos(self.vessel_mA2.ap_a.wall_dist_mask, self.vessel_mA2.ap_a.pixel_positions_I_BEW, self.vessel_mA2.ap_a.left_triangle,self.vessel_mA2.ap_a.right_triangle)
        self.vessel_mA2.ap_a.set_image_mask(image_mask)
        self.vessel_mA2.ap_a.set_pixel_positions_masked(pixel_pos_masked)    

        image_mask, pixel_pos_masked =make_final_mask_and_pixel_pos(self.vessel_mA2.as_a.wall_dist_mask, self.vessel_mA2.as_a.pixel_positions_I_BEW, self.vessel_mA2.as_a.left_triangle, self.vessel_mA2.as_a.right_triangle)
        self.vessel_mA2.as_a.set_image_mask(image_mask)
        self.vessel_mA2.as_a.set_pixel_positions_masked(pixel_pos_masked)  

        image_mask, pixel_pos_masked =make_final_mask_and_pixel_pos(self.vessel_mA2.as_s.wall_dist_mask, self.vessel_mA2.as_s.pixel_positions_I_BEW, self.vessel_mA2.as_s.left_triangle, self.vessel_mA2.as_s.right_triangle)
        self.vessel_mA2.as_s.set_image_mask(image_mask)
        self.vessel_mA2.as_s.set_pixel_positions_masked(pixel_pos_masked)  
        
        
        self.subscriptions = []
        self.images_recieved = [None]*8
        

        for topic_name in topic_names:
            camera_name = topic_name[15:]
            print(camera_name)
            
            
            sub = self.create_subscription(Image, topic_name,self.image_callback,
                                           QoSProfile(depth=10,
                                            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                                            reliability=QoSReliabilityPolicy.BEST_EFFORT))
            self.subscriptions.append(sub)

        self.publisher = self.create_publisher(
            Image,
            'BEW_im',
            QoSProfile(depth=10,
                       durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                       reliability=QoSReliabilityPolicy.RELIABLE))
        self.bridge = CvBridge()    
        
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

    def image_callback(self, msg):
        # Extract the camera index from the topic name
        id = msg.header.frame_id
        # print(id)
        # id = topic_name[16:]
        # print('Recieved image from: '+str(id))

        if(id =='fp_p'):
            self.vessel_mA2.fp_p._im = self.msg_to_cv(msg)
            self.images_recieved[0] = 1

        elif(id =='fp_f'):
            self.vessel_mA2.fp_f._im = self.msg_to_cv(msg)
            self.images_recieved[1] = 1
            
        elif(id =='fs_f'):
            self.vessel_mA2.fs_f._im = self.msg_to_cv(msg)
            self.images_recieved[2] = 1

        elif(id =='fs_s'):
            self.vessel_mA2.fs_s._im = self.msg_to_cv(msg)
            self.images_recieved[3] = 1

        elif(id =='ap_p'):
            self.vessel_mA2.ap_p._im = self.msg_to_cv(msg)
            self.images_recieved[4] = 1

        elif(id =='ap_a'):
            self.vessel_mA2.ap_a._im = self.msg_to_cv(msg)
            self.images_recieved[5] = 1

        elif(id =='as_a'):
            self.vessel_mA2.as_a._im = self.msg_to_cv(msg)
            self.images_recieved[6] = 1

        elif(id =='as_s'):
            self.vessel_mA2.as_s._im = self.msg_to_cv(msg)
            self.images_recieved[7] = 1
        

        # If we have received images from all 8 cameras, combine them and publish the result
        if all(self.images_recieved):
            
            BEW_im = make_BEW(self.vessel_mA2)
            self.images_recieved = [None]*8
            BEW_msg = self.cv_to_msg(BEW_im)
            self.publisher.publish(BEW_msg)
            


def main(args=None):
    
    rclpy.init(args=args)
    # pr = cProfile.Profile()
    # pr.enable()
    bew_image = BEWImage()
    # pr.disable()
    # stats = pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(10)
    # stats.print_stats('cumtime')

    pr = cProfile.Profile()
    pr.enable()
    
    try:
        while rclpy.ok():
            
            rclpy.spin(bew_image)
            
    
    except KeyboardInterrupt:
        pass
    finally:
        bew_image.destroy_node()
        rclpy.shutdown()   
    
    pr.disable()
    stats = pstats.Stats(pr)
    stats.dump_stats(filename='/home/mathias/Documents/ros2_ws_master/src/py_undist/py_undist/Profiler_stats/test_first_attempt_ros2_node1200x1200.prof') # prøv med absoultt file path
    
    # try:
    #     while rclpy.ok():
    #         rclpy.spin_once(bew_image)
    #         user_input = input("Press 'q' to stop spinning: ")
    #         if user_input == 'q':
    #             break
                
    # finally:
    #     bew_image.destroy_node()
    #     rclpy.shutdown()
    print('Exiting main()')

if __name__ == '__main__':
    main()