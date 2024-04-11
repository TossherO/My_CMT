import json
import yaml
from concurrent import futures as futures
import mmengine
import argparse
import numpy as np


def options():
    parser = argparse.ArgumentParser(description='JRDB converting ...')
    parser.add_argument('--path_root',type=str,default='./data/JRDB/')
    parser.add_argument('--split_file',type=str,default='split.json')
    parser.add_argument('--split',type=str,default='train')
    args = parser.parse_args()
    return args


def load_file(path_root, scene_file):

    with open(path_root + '/calibration/lidars.yaml', 'r') as f:
        calib = yaml.safe_load(f)
    cams_calib = {}
    for cam_id in ['0', '2', '4', '6', '8']:
        cam_calib = calib['sensor_'+cam_id]
        distored_K = np.array(cam_calib['distorted_img_K']).reshape([3, 3])
        D = np.array(cam_calib['D'])
        undistorted_K = np.array(cam_calib['undistorted_img_K']).reshape([3, 3])
        temp = np.eye(4)
        temp[:3, :3] = undistorted_K
        undistorted_K = temp
        lidar2cam = np.linalg.inv(np.array(cam_calib['cam2ego']))
        cams_calib['cam'+cam_id] = {'undistorted_K': undistorted_K, 'distored_K': distored_K, 'D': D, 'lidar2cam': lidar2cam}
    
    with open(path_root + 'labels/labels_3d/' + scene_file + '.json', 'r') as load_f:
        labels_3d = json.load(load_f)['labels']
    labels_2d = {}
    for image in ['0', '2', '4', '6', '8']:
        with open(path_root + 'labels/labels_2d/' + scene_file + '_image' + image + '.json', 'r') as load_f:
            labels_2d['cam'+image] = json.load(load_f)['labels']

    def process_single_frame(idx):
        print(scene_file, idx)
        images = {}
        for image in ['0', '2', '4', '6', '8']:
            images['cam'+image] = {}
            images['cam'+image]['img_path'] = 'images/image_' + image + '/' + scene_file + '/' + str(idx).zfill(6) + '.jpg'
            images['cam'+image]['height'] = 480
            images['cam'+image]['width'] = 752
            images['cam'+image]['cam2img'] = cams_calib['cam'+image]['undistorted_K'].tolist()
            images['cam'+image]['lidar2cam'] = cams_calib['cam'+image]['lidar2cam'].tolist()
            images['cam'+image]['lidar2img'] = (cams_calib['cam'+image]['undistorted_K'] @ cams_calib['cam'+image]['lidar2cam']).tolist()
            images['cam'+image]['distored_K'] = cams_calib['cam'+image]['distored_K'].tolist()
            images['cam'+image]['D'] = cams_calib['cam'+image]['D']

        lidar_points = {}
        lidar_points['num_pts_feats'] = 4
        lidar_points['lidar_path_upper'] = 'pointclouds/upper_velodyne/' + scene_file + '/' + str(idx).zfill(6) + '.pcd'
        lidar_points['lidar_path_lower'] = 'pointclouds/lower_velodyne/' + scene_file + '/' + str(idx).zfill(6) + '.pcd'
        lidar_points['lower2upper'] = calib['lidar']['lower2upper']
        lidar_points['lidar2ego'] = calib['lidar']['upper2ego']

        instances = []
        for i in range(len(labels_3d[str(idx).zfill(6)+'.pcd'])):
            # if labels_3d[str(idx).zfill(6)+'.pcd'][i]['box']['h'] < 1.2: # delete the sitting person currently 
            #     continue
            id = labels_3d[str(idx).zfill(6)+'.pcd'][i]['label_id']
            instance = {}
            cam = -1
            instance['occluded'] = 'None'
            instance['truncated'] = 'None'
            for cam_id in ['0', '2', '4', '6', '8']:
                for item in labels_2d['cam'+cam_id][str(idx).zfill(6)+'.jpg']:
                    if item['label_id'] == id:
                        x, y, w, h = item['box']
                        cam = int(cam_id)
                        try:
                            instance['occluded'] = item['attributes']['occlusion']
                            instance['truncated'] = item['attributes']['truncated']
                        except:
                            print('no occluded or truncated')
                        break
                if cam != -1:
                    break
            else:
                x, y, w, h = -1, -1, 0, 0
            instance['bbox'] = [x, y, x+w, y+h]
            instance['bbox_label'] = 0
            instance['center_2d'] = [x-w/2, y-h/2]
            instance['cam'] = cam
            item = labels_3d[str(idx).zfill(6)+'.pcd'][i]
            x, y, z = item['box']['cx'], item['box']['cy'], item['box']['cz']
            l, w, h = item['box']['l'], item['box']['w'], item['box']['h']
            yaw = item['box']['rot_z']
            instance['bbox_3d'] = [x, y, z-h/2, l, w, h, yaw]
            instance['bbox_label_3d'] = 0
            instance['depth'] = item['attributes']['distance']
            instance['num_lidar_pts'] = item['attributes']['num_points']
            instance['difficulty'] = 0
            instance['group_id'] = 0
            instances.append(instance) 
        return {'sample_idx':idx, 'images':images, 'lidar_points':lidar_points, 'instances':instances}
    
    ids = list(range(len(labels_3d)))
    with futures.ThreadPoolExecutor(1) as executor:
        infos = executor.map(process_single_frame, ids)
    return list(infos)


def create_data_info(path_root, scene_file_list):
    data_list = []
    for scene_file in scene_file_list:
        info = load_file(path_root, scene_file)    
        if info:
            data_list = data_list + info
    metainfo = {'categories': {'Pedestrian': 0}, 'dataset': 'JRDB', 'info_version': '1.0'}
    return {'metainfo': metainfo, 'data_list': data_list}


def main():
    args = options()
    path_root = args.path_root
    split = args.split
    split_file = args.path_root + args.split_file
    with open(split_file, 'r') as load_f:
        load_dict = json.load(load_f)
    # info = create_data_info(path_root=path_root, scene_file_list=load_dict[split])
    # filename = f'{path_root}JRDB_infos_{split}.pkl'
    info = create_data_info(path_root=path_root, scene_file_list=load_dict[split][3:5])
    filename = f'{path_root}JRDB_infos_{split}_small.pkl'
    print(f'dataset info {split} file is saved to {filename}')
    mmengine.dump(info, filename)


if __name__ == "__main__":
    main()