from py_undist.calculate_bew_data import calculate_BEW_points_and_mask, calculate_BEW_points_and_rgb_for_interpolation
from py_undist.cls_Camera import Camera
from py_undist.file_handling import try_load_vtx_wts
from py_undist.get_black_fill_pos_rgb import get_black_pixel_pos_and_rgb
import numpy as np
from scipy.spatial import Delaunay
import cv2
import matplotlib.pylab as plt
from py_undist.geometry import line_segment_intersection


class mA2:    

    def __init__(self,
                 camera_fp_p: Camera,
                 camera_fp_f: Camera,
                 camera_fs_f: Camera,
                 camera_fs_s: Camera,
                 camera_ap_p: Camera,
                 camera_ap_a: Camera,
                 camera_as_f: Camera,
                 camera_as_s: Camera):
        self._fp_p = camera_fp_p        
        self._fp_f = camera_fp_f
        self._fs_f = camera_fs_f
        self._fs_s = camera_fs_s
        self._ap_p = camera_ap_p
        self._ap_a = camera_ap_a
        self._as_a = camera_as_f
        self._as_s = camera_as_s
        self._black_pixel_pos, self._black_pixel_rgb = get_black_pixel_pos_and_rgb()
        self._vtx, self._wts  = try_load_vtx_wts()
        # self._delaunay = self.init_delaunay()

    
    def set_vtx(self, vtx):
        self._vtx = vtx
    def set_wts(self, wts):
        self._wts = wts

    def init_delaunay(self):
        # points_fp_f, _ = calculate_BEW_points_and_mask(self.fp_f.wall_mask, cv2.imread('Images/Delaunay_init_images/delaunay_init_im_fp_f.png'))
        # points_fs_f, _ = calculate_BEW_points_and_mask(self.fs_f.wall_mask, cv2.imread('Images/Delaunay_init_images/delaunay_init_im_fs_f.png'))
        # points_fs_s, _ = calculate_BEW_points_and_mask(self.fs_s.wall_mask, cv2.imread('Images/Delaunay_init_images/delaunay_init_im_fs_s.png'))
        # points_ap_p, _ = calculate_BEW_points_and_mask(self.ap_p.wall_mask, cv2.imread('Images/Delaunay_init_images/delaunay_init_im_ap_p.png'))
        # points_ap_a, _ = calculate_BEW_points_and_mask(self.ap_a.wall_mask, cv2.imread('Images/Delaunay_init_images/delaunay_init_im_ap_a.png'))
        # points_as_a, _ = calculate_BEW_points_and_mask(self.as_a.wall_mask, cv2.imread('Images/Delaunay_init_images/delaunay_init_im_as_a.png'))
        # points_as_s, _ = calculate_BEW_points_and_mask(self.as_s.wall_mask, cv2.imread('Images/Delaunay_init_images/delaunay_init_im_as_s.png'))
        points = np.vstack((self.fp_p.pixel_positions_masked,
                            self.fp_f.pixel_positions_masked,
                            self.fs_f.pixel_positions_masked,
                            self.fs_s.pixel_positions_masked,
                            self.ap_p.pixel_positions_masked,
                            self.ap_a.pixel_positions_masked,
                            self.as_a.pixel_positions_masked,
                            self.as_s.pixel_positions_masked,
                            self.black_pixel_pos))
        
        return Delaunay(points)

    def set_points_in_triangle_between_two_cameras(self,camera_left: Camera, camera_right: Camera):
        corner_right_furthest = camera_left.corners[3]
        corner_right_closest = camera_left.corners[2]
        corner_left_furthest =camera_right.corners[0]
        corner_left_closest = camera_right.corners[1]
        

        intersection_point = line_segment_intersection(corner_left_closest,corner_left_furthest,corner_right_closest,corner_right_furthest)
        mid_point_furthest_line = np.array([(corner_left_furthest[0]+ corner_right_furthest[0])/2,
                                        (corner_left_furthest[1]+ corner_right_furthest[1])/2])

        camera_right.set_left_triangle(np.array([intersection_point, mid_point_furthest_line, corner_left_furthest]))
        camera_left.set_right_triangle(np.array([intersection_point, mid_point_furthest_line, corner_right_furthest]))
        
    def find_triangle_between_each_cameras(self):
        # corners = furthest_left,closest_left, closest_right, furthest_right
        # (y down, x right)
        # fp_f x fs_f
        # fp_f_right_furthest = self.fp_f.corners[3]
        # fp_f_right_closest = self.fp_f.corners[2]
        # fs_f_left_furthest =self.fs_f.corners[0]
        # fs_f_left_closest = self.fs_f.corners[1]
        # fp_f_right_furthest = np.array([1005,0])
        # fp_f_right_closest = np.array([727,362])
        # fs_f_left_furthest = np.array([444,0])
        # fs_f_left_closest = np.array([803,370])
        # fp_f_right_furthest = np.array([0, 1026])
        # fp_f_right_closest = np.array([362,727])
        # fs_f_left_furthest = np.array([0,444])
        # fs_f_left_closest = np.array([370,803])

        
        # intersection_point = line_segment_intersection(fs_f_left_closest,fs_f_left_furthest,fp_f_right_closest,fp_f_right_furthest).reshape(2)
        
        # mid_point_furthest_line = np.array([(fs_f_left_furthest[0]+ fp_f_right_furthest[0])/2,
        #                                 (fs_f_left_furthest[1]+ fp_f_right_furthest[1])/2])

        # mid_point_furthest_line_inside_image = line_segment_intersection(intersection_point,mid_point_furthest_line,image_furthest_side[0],image_furthest_side[1]).reshape(2)
        # fs_f_left_furthest_inside_image = line_segment_intersection(fs_f_left_closest,fs_f_left_furthest,image_furthest_side[0],image_furthest_side[1]).reshape(2)
        
        # fp_f_furthest_right_inside_image = line_segment_intersection(fp_f_right_closest,fp_f_right_furthest,image_furthest_side[0],image_furthest_side[1]).reshape(2)
        # mid_point_furthest_line_inside_image = (np.abs(fs_f_left_furthest_inside_image)+np.abs(fp_f_furthest_right_inside_image))/2

        # self.fs_f.set_left_triangle(np.array([intersection_point, mid_point_furthest_line, fs_f_left_furthest]))
        # self.fp_f.set_right_triangle(np.array([intersection_point, mid_point_furthest_line, fp_f_right_furthest]))
        
        # plt.figure()
        # pixels = self.ap_a.pixel_positions_I_BEW[0,:,:]
        # plt.imshow(pixels)
        # plt.show()

        # fs_f x fs_s #
        self.set_points_in_triangle_between_two_cameras(self.fp_p, self.fp_f)
        self.set_points_in_triangle_between_two_cameras(self.fp_f, self.fs_f)
        self.set_points_in_triangle_between_two_cameras(self.fs_f, self.fs_s)
        # self.set_points_in_triangle_between_two_cameras(self.ap_p, self.ap_a)
        # self.set_points_in_triangle_between_two_cameras(self.ap_a, self.as_a)    
        # self.set_points_in_triangle_between_two_cameras(self.ap_a, self.as_s)

        # ap_p x ap_a # 
        # ap_p_right_furthest = self.ap_p.corners[3]
        # ap_p_right_closest = self.ap_p.corners[2]
        # ap_a_left_furthest =self.ap_a.corners[0]
        # ap_a_left_closest = self.ap_a.corners[1]
        ap_p_right_furthest = np.array([2000,700])
        ap_p_right_closest = np.array([1121,620])
        ap_a_left_furthest =np.array([1900,-500])
        ap_a_left_closest = np.array([1130,630])

        intersection_point = line_segment_intersection(ap_a_left_closest,ap_a_left_furthest,ap_p_right_closest,ap_p_right_furthest)
        mid_point_furthest_line = np.array([(ap_a_left_furthest[0]+ ap_p_right_furthest[0])/2,
                                        (ap_a_left_furthest[1]+ ap_p_right_furthest[1])/2])

        self.ap_a.set_left_triangle(np.array([intersection_point, mid_point_furthest_line, ap_a_left_furthest]))
        self.ap_p.set_right_triangle(np.array([intersection_point, mid_point_furthest_line, ap_p_right_furthest]))

        # as_a x as_s #
        # as_a_right_furthest = self.as_a.corners[3]
        # as_a_right_closest = self.as_a.corners[2]
        # as_s_left_furthest =self.as_s.corners[0]
        # as_s_left_closest = self.as_s.corners[1]
        as_a_right_furthest = np.array([1501,1501])
        as_a_right_closest = np.array([1120,870])
        as_s_left_furthest =np.array([1501,980])
        as_s_left_closest = np.array([1100,880])

        intersection_point = line_segment_intersection(as_s_left_furthest,as_s_left_closest,as_a_right_closest,as_a_right_furthest)
        mid_point_furthest_line = np.array([(as_s_left_furthest[0]+ as_a_right_furthest[0])/2,
                                        (as_s_left_furthest[1]+ as_a_right_furthest[1])/2])

        self.as_s.set_left_triangle(np.array([intersection_point, mid_point_furthest_line, as_s_left_furthest]))
        self.as_a.set_right_triangle(np.array([intersection_point, mid_point_furthest_line, as_a_right_furthest]))

        # ap_a x as_a
        ap_a_right_furthest = self.ap_a.corners[3]
        ap_a_right_closest = self.ap_a.corners[2]
        as_a_left_furthest = np.array([1501,-200])
        as_a_left_closest = self.as_a.corners[1]


        intersection_point_a = line_segment_intersection(as_a_left_furthest, as_a_left_closest, ap_a_right_furthest,ap_a_right_closest)
        mid_point_furthest_line_a = np.array([(as_a_left_furthest[0]+ ap_a_right_furthest[0])/2,
                                             (as_a_left_furthest[1]+ ap_a_right_furthest[1])/2])

        self.as_a.set_left_triangle(np.array([intersection_point_a, mid_point_furthest_line_a, as_a_left_furthest]))
        self.ap_a.set_right_triangle(np.array([intersection_point_a, mid_point_furthest_line_a, ap_a_right_furthest]))
        pass

    @property
    def fp_p(self):
        return self._fp_p
    @property
    def fp_f(self):
        return self._fp_f
    @property
    def fs_f(self):
        return self._fs_f
    @property
    def fs_s(self):
        return self._fs_s
    @property
    def ap_p(self):
        return self._ap_p
    @property
    def ap_a(self):
        return self._ap_a
    @property
    def as_a(self):
        return self._as_a
    @property
    def as_s(self):
        return self._as_s
    # @property
    # def delaunay(self):
    #     return self._delaunay
    @property
    def vtx(self):
        return self._vtx
    @property
    def wts(self):
        return self._wts
    @property
    def black_pixel_pos(self):
        return self._black_pixel_pos
    @property
    def black_pixel_rgb(self):
        return self._black_pixel_rgb