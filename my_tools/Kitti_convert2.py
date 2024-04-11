import json
from concurrent import futures as futures
from pathlib import Path
import mmengine
import numpy as np
from mmdet3d.structures import LiDARInstance3DBoxes, CameraInstance3DBoxes


def main():

    info = mmengine.load('./data/kitti/kitti_infos_trainval.pkl')
    filename = './data/kitti/kitti_infos_trainval2.pkl'
    metainfo = {'categories': {'Pedestrian': 0}, 'dataset': 'Kitti', 'info_version': '1.0'}
    count = 0
    
    new_data_list = []
    for data in info['data_list']:
        images = {}
        images['CAM2'] = data['images']['CAM2']
        images['CAM2']['img_path'] = 'training/image_2/' + images['CAM2']['img_path']
        
        lidar_points = {}
        lidar_points['num_pts_feats'] = 4
        lidar_points['lidar_path'] = 'training/velodyne_reduced/' + data['lidar_points']['lidar_path']

        instances = []
        for instance in data['instances']:
            if instance['bbox_label_3d'] == 0:
                lidar2cam = np.array(images['CAM2']['lidar2cam'])
                instance['bbox_3d'] = CameraInstance3DBoxes([instance['bbox_3d']]).convert_to(0, np.linalg.inv(lidar2cam)).tensor[0].tolist()
                instances.append(instance)

        if len(instances) > 0:
            new_data_list.append({'sample_idx':data['sample_idx'], 'images':images, 'lidar_points':lidar_points, 'instances':instances})
        count += 1
        if count % 100 == 0:
            print(count)

    mmengine.dump({'metainfo': metainfo, 'data_list': new_data_list}, filename)


if __name__ == "__main__":
    main()