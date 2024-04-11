import json
from concurrent import futures as futures
from pathlib import Path
import mmengine
import numpy as np
from mmdet3d.structures import LiDARInstance3DBoxes, CameraInstance3DBoxes


def main():

    info1 = mmengine.load('./data/kitti/kitti_infos_trainval2.pkl')
    info2 = mmengine.load('./data/STCrowd/STCrowd_infos_train.pkl')
    filename = './data/kitti_stc_fusion_infos_train.pkl'
    metainfo = {'categories': {'Pedestrian': 0}, 'dataset': 'Kitti_STCrowd', 'info_version': '1.0'}
    count = 0
    new_data_list = []

    for data in info1['data_list']:
        images = {}
        images['left'] = data['images']['CAM2']
        images['left']['img_path'] = 'kitti/' + images['left']['img_path']
        lidar_points = {}
        lidar_points['num_pts_feats'] = 4
        lidar_points['lidar_path'] = 'kitti/' + data['lidar_points']['lidar_path']
        instances = []
        for instance in data['instances']:
            x, y, z = instance['bbox_3d'][:3]
            if 0 <= x <= 36 and -18 <= y <= 18 and -5 <= z <= 1:
                difficult = instance['occluded']
                dist = np.sqrt(x**2 + y**2)
                if dist > 20:
                    difficult += 2
                elif dist > 10:
                    difficult += 1
                instance['difficulty'] = difficult // 2
                instances.append(instance)
        if len(instances) > 0:
            new_data_list.append({'sample_idx':count, 'images':images, 'lidar_points':lidar_points, 'instances':instances})
            count += 1
        if count % 100 == 0:
            print(count)
    
    for data in info2['data_list']:
        images = {}
        images['left'] = data['images']['left']
        images['left']['img_path'] = 'STCrowd/' + images['left']['img_path']
        lidar_points = {}
        lidar_points['num_pts_feats'] = 4
        lidar_points['lidar_path'] = 'STCrowd/' + data['lidar_points']['lidar_path']
        instances = []
        for instance in data['instances']:
            x, y, z = instance['bbox_3d'][:3]
            if 0 <= x <= 36 and -18 <= y <= 18 and -5 <= z <= 1:
                difficult = instance['occluded']
                dist = np.sqrt(x**2 + y**2)
                if dist > 20:
                    difficult += 2
                elif dist > 10:
                    difficult += 1
                instance['difficulty'] = difficult // 2
                instances.append(instance)
        if len(instances) > 0:
            new_data_list.append({'sample_idx':count, 'images':images, 'lidar_points':lidar_points, 'instances':instances})
            count += 1
        if count % 100 == 0:
            print(count)

    mmengine.dump({'metainfo': metainfo, 'data_list': new_data_list}, filename)


if __name__ == "__main__":
    main()