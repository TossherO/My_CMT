import json
import yaml
from concurrent import futures as futures
import mmengine
import argparse
import numpy as np


def options():
    parser = argparse.ArgumentParser(description='JRDB converting ...')
    parser.add_argument('--path_root',type=str,default='./data/SiT/')
    parser.add_argument('--split_file',type=str,default='split.json')
    parser.add_argument('--split',type=str,default='train')
    args = parser.parse_args()
    return args


def get_pts_in_3dbox_(pc, corners):
    num_pts_in_gt = []
    for num, corner in enumerate(corners):
        x_max, x_min = corner[:, 0].max(), corner[:, 0].min()
        y_max, y_min = corner[:, 1].max(), corner[:, 1].min()
        z_max, z_min = corner[:, 2].max(), corner[:, 2].min()

        mask_x = np.logical_and(pc[:,0] >= x_min, pc[:, 0] <= x_max)
        mask_y = np.logical_and(pc[:,1] >= y_min, pc[:, 1] <= y_max)
        mask_z = np.logical_and(pc[:,2] >= z_min, pc[:, 2] <= z_max)
        mask = mask_x * mask_y * mask_z
        num_pts_in_gt.append(mask.sum())

    return num_pts_in_gt


def box_center_to_corner_3d_(box_center):
    # To return
    translation = box_center[0:3]
    l, w, h = box_center[3], box_center[4], box_center[5]
    rotation = box_center[6]

    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2] #[0, 0, 0, 0, -h, -h, -h, -h]
    # z_corners = [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]
    bounding_box = np.vstack([x_corners, y_corners, z_corners])

    rotation_matrix = np.array([[np.cos(rotation),  -np.sin(rotation), 0],
                                [np.sin(rotation), np.cos(rotation), 0],
                                [0,  0,  1]])

    corner_box = (np.dot(rotation_matrix, bounding_box).T + translation).T

    return corner_box


def load_file(path_root, scene_file, idx):

    def process_single_frame(idx):

        print(scene_file, idx)
        
        with open(path_root + scene_file + '/calib/' + str(idx%200) + '.txt') as f:
            calib = f.readlines()
        intrinsics = []
        extrinsics = []
        for i in range(5):
            intrinsics.append(np.array([float(info) for info in calib[i].strip().split(' ')[1:10]]).reshape([3, 3]))
        for i in range(5, 10):
            extrinsics.append(np.array([float(info) for info in calib[i].strip().split(' ')[1:13]]).reshape([3, 4]))
        R0_rect = np.array([float(info) for info in calib[10].strip().split(' ')[1:10]]).reshape([3, 3])

        images = {}
        for i in range(5):
            images['P'+str(i)] = {}
            images['P'+str(i)]['img_path'] = scene_file + '/cam_img/' + str(i+1) + '/data_blur/' + str(idx%200) + '.png'
            images['P'+str(i)]['height'] = 1200
            images['P'+str(i)]['width'] = 1920
            temp = np.eye(4)
            temp[:3, :3] = intrinsics[i]
            images['P'+str(i)]['cam2img'] = temp.tolist()
            images['P'+str(i)]['lidar2cam'] = np.concatenate([extrinsics[i], np.array([[0, 0, 0, 1]])], axis=0).tolist()
            images['P'+str(i)]['lidar2img'] = (np.array(images['P'+str(i)]['cam2img']) @ np.array(images['P'+str(i)]['lidar2cam'])).tolist()
        images['R0_rect'] = R0_rect.tolist()

        lidar_points = {}
        lidar_points['num_pts_feats'] = 4
        lidar_points['lidar_path'] = scene_file + '/velo/bin/data/' + str(idx%200) + '.bin'

        points = np.fromfile(path_root + lidar_points['lidar_path'], dtype=np.float32).reshape([-1, 4]) 
        with open(path_root + scene_file + '/label_3d/' + str(idx%200) + '.txt') as f:
            labels_3d = f.readlines()
        labels_3d = [label_3d.strip().split(' ') for label_3d in labels_3d]
        instances = []
        for label_3d in labels_3d:
            instance = {}
            h, l, w = [float(item) for item in label_3d[2:5]]
            x, y, z = [float(item) for item in label_3d[5:8]]
            yaw = float(label_3d[8])
            instance['bbox_3d'] = [x, y, z-h/2, l, w, h, -(-yaw + np.pi/2)]
            instance['bbox_label_3d'] = 0
            instance['depth'] = np.sqrt(x**2 + y**2 + z**2)
            corners_3d = box_center_to_corner_3d_([x, y, z, l, w, h, yaw])
            instance['num_lidar_pts'] = get_pts_in_3dbox_(points, np.expand_dims(corners_3d, axis=0))[0]
            instances.append(instance) 
        return {'sample_idx':idx, 'images':images, 'lidar_points':lidar_points, 'instances':instances}
    
    ids = list(range(idx*200, (idx+1)*200))
    with futures.ThreadPoolExecutor(1) as executor:
        infos = executor.map(process_single_frame, ids)
    return list(infos)


def create_data_info(path_root, scene_file_list):
    data_list = []
    for idx, scene_file in enumerate(scene_file_list):
        info = load_file(path_root, scene_file, idx)    
        if info:
            data_list = data_list + info
    metainfo = {'categories': {'Pedestrian': 0}, 'dataset': 'SiT', 'info_version': '1.0'}
    return {'metainfo': metainfo, 'data_list': data_list}


def main():
    args = options()
    path_root = args.path_root
    split = args.split
    split_file = args.path_root + args.split_file
    with open(split_file, 'r') as load_f:
        load_dict = json.load(load_f)
    info = create_data_info(path_root=path_root, scene_file_list=load_dict[split])
    filename = f'{path_root}SiT_infos_{split}.pkl'
    print(f'dataset info {split} file is saved to {filename}')
    mmengine.dump(info, filename)


if __name__ == "__main__":
    main()