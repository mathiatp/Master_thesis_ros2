import yaml
from yaml.loader import SafeLoader
import numpy as np



def yaml_file_to_dict(file_name: str):
    with open(file_name) as f:
        data = yaml.load(f, Loader=SafeLoader)
    return data

def mA2_frame_yaml_to_rotation_translation_vec(camera_name: str ):
    data_dict = yaml_file_to_dict('/home/mathias/Documents/Master_Thesis/Rosbags/ma2_frames.yml')
    translation = np.array([[data_dict[camera_name]['static_transform_xyz']['x'],
                             data_dict[camera_name]['static_transform_xyz']['y'],
                             data_dict[camera_name]['static_transform_xyz']['z']]])
    rotation = np.array([data_dict[camera_name]['static_transform_xyz']['roll'],
                         data_dict[camera_name]['static_transform_xyz']['pitch'],
                         data_dict[camera_name]['static_transform_xyz']['yaw']])
    return rotation, translation

def camera_name_calib_yaml_to_K_D(camera_name: str):
    data_dict = yaml_file_to_dict('/home/mathias/Documents/ros2_ws_master/src/py_undist/py_undist/calibration_files/'+ camera_name + '_calib.yaml')
    K = np.array([data_dict['camera_matrix']['data']])
    K = K.reshape(data_dict['camera_matrix']['rows'], data_dict['camera_matrix']['cols'])
    D = np.array([data_dict['distortion_coefficients']['data']])
    return K, D

def try_load_vtx_wts():
    try:
        vtx = np.load('/home/mathias/Documents/ros2_ws_master/src/py_undist/py_undist/vtx.npy')
        wts = np.load('/home/mathias/Documents/ros2_ws_master/src/py_undist/py_undist/wts.npy')
        print('load')
        return vtx, wts
    except OSError:
        return None, None