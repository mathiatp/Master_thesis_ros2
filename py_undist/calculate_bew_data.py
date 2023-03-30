import numpy as np
# import cv2
from scipy.spatial.transform import Rotation as R
# from cls_Camera import Camera
from py_undist.config import MAX_DIST_X, MIN_DIST_X, MAX_DIST_Y, MIN_DIST_Y, BEW_IMAGE_HEIGHT, BEW_IMAGE_WIDTH, ROW_CUTOFF
# from scipy.interpolate import griddata
from scipy.spatial import Delaunay
from py_undist.geometry import check_if_point_inside_triangle
# from get_black_fill_pos_rgb import im_mask_walls
# import matplotlib.pyplot as plt

def georeference_point_eq(intrinsic_matrix: np.ndarray,
                            image_points: np.ndarray,
                            camer_rotation: np.ndarray,
                            translation: np.ndarray,
                            target_elevation: np.ndarray) -> np.ndarray:
    """
    Estimate origin of pixel point using georeferencing.
    Relies on as of yet unported ROS2 functionality (transform).
    :param Header header: Header with timestamp
    :param np.ndarray image_points: Image point to georeference
    :param str camera: Origin camera
    :param np.ndarray ownship_elevation: Elevation of ownship
    :return np.ndarray: Cartesian estimate of pixel origin
    """
    intrinsic_matrix = intrinsic_matrix

    rot_mat = R.from_euler('xyz', camer_rotation).as_matrix().T
    t_vec= -rot_mat@np.transpose(translation)
    extrinsic_matrix = np.concatenate((rot_mat, t_vec), axis=1) #This is kind of invR|-invR*t
    
    P = intrinsic_matrix @ extrinsic_matrix

    x_p, y_p = image_points
    # Calculate coefficients for the left/right side of reverse pinhole model
    left_side = np.array(
        [[x_p*P[2, 0] - P[0, 0], x_p*P[2, 1] - P[0, 1]],
            [y_p*P[2, 0] - P[1, 0], y_p*P[2, 1] - P[1, 1]]])

    right_side = np.array(
        [[target_elevation*(P[0, 2]-x_p*P[2, 2])+P[0, 3]-x_p*P[2, 3]],
            [target_elevation*(P[1, 2]-y_p*P[2, 2])+P[1, 3]-y_p*P[2, 3]]])

    xy = np.linalg.inv(left_side)@right_side

    pos_estimate = np.array(
        [xy[0, 0],
            xy[1, 0],
            target_elevation])
    return pos_estimate 

def interp_weights(xy, uv,d=2):
    tri = Delaunay(xy)
    simplex = tri.find_simplex(uv)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uv - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

def interpolate(values, vtx, wts):
    values_id = values[vtx]
    return np.einsum('nj,nj->n', values_id, wts) #row-wise dot product of two matrices then sum the row to one value

def calculate_im_pos(height, width, K, camera_rotation, camera_translation, name):
    try:
        file_name_pixel_position = 'resource/pixel_position_arrays/pixel_position_'+name+'.npy'
        im_pos = np.load(file_name_pixel_position)
        return im_pos
    
    except OSError:
        print('Could not find file: ' + file_name_pixel_position + '. \nStarting to calculate pixel positions for camera ' + name + '.')

        target_elevation = np.array(0)

        xx,yy = np.meshgrid(range(width), range(height), indexing='xy')
        grid = np.array([xx,yy])
        grid = np.einsum('ijk->jki', grid)

        im_pos = np.zeros((height, width,3), dtype=np.float32)

        for row in grid:
            for coor in row:
                pixel = coor
                pos = georeference_point_eq(K, pixel, camera_rotation, camera_translation, target_elevation).astype(np.float32)
                im_pos[int(pixel[1]),int(pixel[0])] = np.concatenate((pos[0:2],np.array([1])))

        np.save(file_name_pixel_position, im_pos)     
        return im_pos

def normalize_im_pos_for_BEW(im_pos_cut_t):
    im_pos_cut_F = im_pos_cut_t
    # Swithing from x forward, y right axis to x right, y forward to match image cooridnates
    switch_xy = np.array([[0,1,0],
                            [1,0,0],
                            [0,0,1]])

    im_pos_cut_F = np.einsum('ij, jkl->ikl', switch_xy, im_pos_cut_F)

    max_x = MAX_DIST_X
    min_x = MIN_DIST_X
    
    max_y = MAX_DIST_Y
    min_y = MIN_DIST_Y

    s_x = 2/(max_x-min_x)
    t_x = -s_x * min_x-1

    s_y = 2/(max_y-min_y)
    t_y = -s_y * min_y-1

    N = np.array([[s_x, 0,  t_x],
                    [0, s_y,  t_y],
                    [0, 0,    1]])
    im_pos_cut_F_normalized = np.einsum('ij,jkl->ikl', N, im_pos_cut_F)

    return im_pos_cut_F_normalized



def calculate_BEW_points_and_rgb_for_interpolation(img_pos, img_undistorted):
    row_cut_off = ROW_CUTOFF
    rgb_chan = 3
    im_pos_cut = img_pos[row_cut_off:,:,:]
    im_pos_cut_T = np.einsum('ijk->kij',im_pos_cut)
    
    
    im_cut = img_undistorted[row_cut_off:,:,:]
    im_pos_normalized = normalize_im_pos_for_BEW(im_pos_cut_T)
    

    new_h, new_w = 1024, 1224

    K = np.array([[new_w/2-1,     0,          new_w/2],
                    [0,           -(new_h/2-1),    new_h/2],
                    [0,           0,          1]])

    im_pos_pixel = np.einsum('ij,jkl->ikl', K, im_pos_normalized)
    
    im_pos_pixel = np.einsum('ijk->jki',im_pos_pixel)
    
    # any_nan = np.isnan(im_pos_pixel).any()
    im_pos_pixel =  np.nan_to_num(im_pos_pixel, nan = 99999999)
    im_pos_pixel = im_pos_pixel.astype(int)
    im_pos_pixel = im_pos_pixel[:,:,:2] #(498, 1224, 2)

    points_x_all = im_pos_pixel[:,:,1] #(498, 1224)
    points_x = np.transpose(np.array([np.ravel(points_x_all)])) #(609552, 1)
    points_x_all = np.transpose(np.array([np.ravel(points_x_all)]))
    points_x = np.transpose(np.array([points_x[points_x != 99999999]]))

    points_y = im_pos_pixel[:,:,0]
    points_y = np.transpose(np.array([np.ravel(points_y)]))
    points_y = np.transpose(np.array([points_y[points_y != 99999999]]))

    # points_y = np.transpose(np.array([np.ravel(im_pos_pixel[im_pos_pixel[:,:,0] != 99999999])]))
    points = np.concatenate((points_x,points_y), axis=1)
    grid_x,grid_y = np.meshgrid(range(new_h), range(new_w), indexing='ij')

    rgb = im_cut

    rgb = np.reshape(rgb,(len(points_x_all), rgb_chan))
    rgb = np.delete(rgb, np.where(points_x_all == 99999999), axis=0)

    # grid_z0 = griddata(points, rgb, (grid_x, grid_y), method='linear')
    # grid_z0[np.where(np.isnan(grid_z0))] = 0
    # grid_z0 = grid_z0[:,:,:].astype(np.uint8)
    
    return  points, rgb 

def calculate_BEW_points_and_mask(img_pos: np.array, wall_mask: np.array):
    
    row_cut_off = ROW_CUTOFF
    rgb_chan = 3
    # img_pos 1024,1224,3
    im_pos_cut = img_pos[row_cut_off:,:,:]  # 498,1224,3
    im_pos_cut_T = np.einsum('ijk->kij',im_pos_cut) # 3,498,1224
    
    wall_mask = wall_mask
    wall_mask = wall_mask[row_cut_off:,:] #498,1224, type=bool
    image_mask = np.array([wall_mask]) # 1, 498,1224
    image_mask[0,:,:][np.where(im_pos_cut_T[0,:,:]> MAX_DIST_X)] = False
    image_mask[0,:,:][np.where(im_pos_cut_T[0,:,:]< MIN_DIST_X)] = False
    image_mask[0,:,:][np.where(im_pos_cut_T[1,:,:]< MIN_DIST_Y)] = False
    image_mask[0,:,:][np.where(im_pos_cut_T[1,:,:]> MAX_DIST_Y)] = False # 1, 498,1224


    max_x = MAX_DIST_X
    min_x = MIN_DIST_X
    
    max_y = MAX_DIST_Y
    min_y = MIN_DIST_Y

    s_x = 2/(max_x-min_x)
    t_x = -s_x * min_x-1

    s_y = 2/(max_y-min_y)
    t_y = -s_y * min_y-1

    Normalizing = np.array([[s_x, 0,  t_x],
                            [0, s_y,  t_y],
                            [0, 0,    1]])

    switch_xy = np.array([[0,1,0],
                            [1,0,0],
                            [0,0,1]])
    N = switch_xy@Normalizing
    im_pos_cut_T = np.einsum('ij,jkl->ikl', switch_xy, im_pos_cut_T)
    im_pos_cut_T_normalized = np.einsum('ij,jkl->ikl', Normalizing, im_pos_cut_T) # 3,498,1224
    
    # check_im_pos_normalized = im_pos_cut_T_normalized
    # check_im_pos_normalized = np.einsum('ijk->jki',check_im_pos_normalized)
    # check_im_pos_normalized = check_im_pos_normalized[:,:,:2]
   
    # check_im_pos_normalized[:,:,:][np.where(image_mask[0,:,:] == False)] = 0
    # x_max = np.max(check_im_pos_normalized[:,:,0])
    # y_max = np.max(check_im_pos_normalized[:,:,1])

    # x_min = np.min(check_im_pos_normalized[:,:,0])
    # y_min = np.min(check_im_pos_normalized[:,:,1])
    

    new_h, new_w = BEW_IMAGE_HEIGHT, BEW_IMAGE_WIDTH

    K = np.array([[new_w/2-1,     0,          new_w/2],
                    [0,           -(new_h/2-1),    new_h/2],
                    [0,           0,          1]])

    im_pos_pixel_I_BEW = np.einsum('ij,jkl->ikl', K, im_pos_cut_T_normalized) # 3,498,1224 dtype=float64
    #same here
    
    return image_mask, im_pos_pixel_I_BEW
    
    
    
    """
    # Test to get the same pixel_pos matrix to match rgb matrix - This is from old method
    # im_pos_pixel_1 = im_pos_pixel
    # im_pos_pixel_1 = np.einsum('ijk->jki',im_pos_pixel_1)
    # im_pos_pixel_1 = im_pos_pixel_1[:,:,:2] #(498, 1224, 2)
    # points_x = im_pos_pixel_1[:,:,1] #(498, 1224)
    # points_x = np.transpose(np.array([np.ravel(points_x)])) #(609552, 1)
    # points_y = im_pos_pixel_1[:,:,0] #(498, 1224)
    # points_y = np.transpose(np.array([np.ravel(points_y)])) #(609552, 1)
    # points = np.concatenate((points_x,points_y), axis=1) #(609552, 2)

    im_pos_pixel = np.einsum('ij,jkl->ikl', switch_xy, im_pos_pixel) # 3,498,1224
    im_pos_pixel = im_pos_pixel[:2,:,:] # 2,498,1224
    im_pos_pixel = np.einsum('ijk->jki',im_pos_pixel)
    im_pos_pixel_2_col = np.reshape(im_pos_pixel,(np.shape(im_pos_pixel)[0]*np.shape(im_pos_pixel)[1], 2)) # 609552, 2

    image_mask_1_col = np.reshape(image_mask,(np.shape(im_pos_pixel_2_col)[0],1)) #(609552, 1)
    image_mask_2_col_stack = np.column_stack((image_mask_1_col,image_mask_1_col)) # (609552, 2)
    image_mask_3_col_stack = np.column_stack((image_mask_1_col,image_mask_1_col,image_mask_1_col)) # (609552, 3)

    # im_pos_pixel = np.einsum('ijk->jk',im_pos_pixel)
    # image_mask = np.einsum('ijk->jk', image_mask)
    
    # true_count = np.count_nonzero(image_mask_2_col_stack[:,0]==True)

    im_pos_true = im_pos_pixel_2_col[np.all(image_mask_2_col_stack==True, axis=1)] #(499350, 2)
    
    
    # im_rgb = img_undistorted[row_cut_off:,:] #(498, 1224, 3)
    # im_rgb_T = np.reshape(im_rgb,(np.shape(im_pos_pixel_2_col)[0], 3)) #(609552, 3)
    # im_rgb_true = im_rgb_T[np.all(image_mask_3_col_stack==True, axis=1)] # (499350, 3)



    # im = img_undistorted[row_cut_off:,:,:]
    # image_mask = np.einsum('ijk->jk', image_mask)
    # im[:,:,:][np.where(image_mask==False)] = np.array([0,0,0])

    # im = np.delete(im,np.where(image_mask==False),axis=0)
    # im = np.reshape(im,(np.shape(im_pos_pixel)[0], 3))
    # print(im==im_rgb_true)
    # plt.figure()
    # plt.imshow(im)
    # plt.show()
    return image_mask_3_col_stack, im_pos_true.astype(int)
    """
def make_final_mask_and_pixel_pos(im_mask, im_pos_pixel, left_triangle, right_triangle):
    image_mask = im_mask 
    
    # im_mask = np.einsum('ijk->jk', im_mask)
    # plt.figure('before triangle')
    # plt.imshow(im_mask)
    mask_true_count = np.count_nonzero(im_mask)
  

    # if (right_triangle is not None):
    #     image_mask[0,:,:][np.where(im_pos_pixel[0,:,:]> 750)] = False
    # mask_true_count_after = np.count_nonzero(image_mask)
    if (right_triangle is not None):
        for row in range(np.shape(im_pos_pixel)[1]):
            for col in range(np.shape(im_pos_pixel)[2]):
                point = np.array([im_pos_pixel[1,row,col],im_pos_pixel[0,row,col]])
                point_validity = not check_if_point_inside_triangle(point,right_triangle)
                a = image_mask[0,row,col]
                check = (image_mask[0,row,col] and point_validity)
                image_mask[0,row,col] = check
    # if (left_triangle is not None):
    #     image_mask[0,:,:][np.where(im_pos_pixel[1,:,:]<left_triangle[0,1])] = False
    if (left_triangle is not None):
        for row in range(np.shape(im_pos_pixel)[1]):
            for col in range(np.shape(im_pos_pixel)[2]):
                point = np.array([im_pos_pixel[1,row,col], im_pos_pixel[0,row,col]])
                point_validity = not check_if_point_inside_triangle(point,left_triangle)
                a = image_mask[0,row,col]
                check = (image_mask[0,row,col] and point_validity)
                image_mask[0,row,col] = check
    mask_true_count_after = np.count_nonzero(image_mask)
    switch_xy = np.array([[0,1,0],
                        [1,0,0],
                        [0,0,1]])

    im_pos_pixel = np.einsum('ij,jkl->ikl', switch_xy, im_pos_pixel) # 3,498,1224

    im_pos_pixel = im_pos_pixel[:2,:,:] # 2,498,1224
    im_pos_pixel = np.einsum('ijk->jki',im_pos_pixel)
    im_pos_pixel_2_col = np.reshape(im_pos_pixel,(np.shape(im_pos_pixel)[0]*np.shape(im_pos_pixel)[1], 2)) # 609552, 2

    image_mask_1_col = np.reshape(image_mask,(np.shape(im_pos_pixel_2_col)[0],1)) #(609552, 1)
    image_mask_2_col_stack = np.column_stack((image_mask_1_col,image_mask_1_col)) # (609552, 2)
    image_mask_3_col_stack = np.column_stack((image_mask_1_col,image_mask_1_col,image_mask_1_col)) # (609552, 3)

    im_pos_true = im_pos_pixel_2_col[np.all(image_mask_2_col_stack==True, axis=1)] #(499350, 2)

    mask_true_count_after = np.count_nonzero(image_mask_1_col)
    image_mask_after = np.einsum('ijk->jk', image_mask)
    # plt.figure('after triangle')
    # plt.imshow(image_mask_after)
    # plt.show()
    
    
    return image_mask_3_col_stack, im_pos_true.astype(int)

def calculate_rgb_matrix_for_BEW(img_undistorted: np.array, image_mask: np.array):
    im_rgb = img_undistorted[ROW_CUTOFF:,:] #(498, 1224, 3)
    im_rgb_T = np.reshape(im_rgb,(np.shape(image_mask)[0], 3)) #(609552, 3)
    
    im_rgb_true = im_rgb_T[image_mask==True]
    im_rgb_true = np.reshape(im_rgb_true,(3,-1),order='F')
    im_rgb_true_T= im_rgb_true.transpose()
    
    return im_rgb_true_T


