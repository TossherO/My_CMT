import json
from concurrent import futures as futures
from pathlib import Path
import mmengine
import argparse
import numpy as np


def options():
    parser = argparse.ArgumentParser(description='STCrowd converting ...')
    parser.add_argument('--path_root',type=str,default='./data/STCrowd/')
    parser.add_argument('--split_file',type=str,default='split.json')   # the split file 
    parser.add_argument('--split',type=str,default='train')             # train / val 
    args = parser.parse_args()
    return args


def load_file(path_root,load_dict):
    # load file from single json file and return a list
    # it is dealing the continuous sequence
    def process_single_scene(idx):
        
        images = {}
        images['left'] = {}
        images['left']['img_path'] = load_dict['frames'][idx]['images'][0]['image_name']
        images['left']['height'] = 720
        images['left']['width'] = 1280
        images['left']['cam2img'] = [[683.8, 0., 673.5907, 0.], [0., 684.147, 372.8048, 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]
        images['left']['lidar2cam'] = [[0.00852965, -0.999945, -0.00606215, 0.0609592],
                                       [-0.0417155, 0.00570127, -0.999113, -0.144364],
                                       [0.999093, 0.00877497, -0.0416646, -0.0731114],
                                       [0., 0., 0., 1.]]
        images['left']['lidar2img'] = np.array(images['left']['cam2img']) @ np.array(images['left']['lidar2cam']).tolist()
        images['R0_rect'] = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]

        lidar_points = {}
        lidar_points['num_pts_feats'] = 4
        lidar_points['lidar_path'] = load_dict['frames'][idx]['frame_name']

        instances = []
        for i in range(len(load_dict['frames'][idx]['items'])):
            if load_dict['frames'][idx]['items'][i]['boundingbox']['z'] < 1.2: # delete the sitting person currently 
                continue
            number = load_dict['frames'][idx]['items'][i]['number']
            instance = {}
            for item in load_dict['frames'][idx]['images'][0]['items']:
                if item['number'] == number:
                    x, y, w, h = item['boundingbox']['x'], item['boundingbox']['y'], item['dimension']['x'], item['dimension']['y']
                    break
            else:
                x, y, w, h = -1, -1, 0, 0
            instance['bbox'] = [x-w/2, y-h/2, x+w/2, y+h/2]
            instance['bbox_label'] = 0
            instance['center_2d'] = [x, y]
            item = load_dict['frames'][idx]['items'][i]
            x, y, z = item['position']['x'], item['position']['y'], item['position']['z']
            l, w, h = item['boundingbox']['x'], item['boundingbox']['y'], item['boundingbox']['z']
            yaw = item['rotation']
            instance['bbox_3d'] = [x, y, z-h/2, l, w, h, yaw]
            instance['bbox_label_3d'] = 0
            instance['depth'] = np.sqrt(x**2 + y**2 + z**2)
            instance['num_lidar_pts'] = item['pointCount']
            instance['difficulty'] = 0
            instance['truncated'] = 0.0
            instance['occluded'] = item['occlusion']
            instance['group_id'] = 0
            instances.append(instance)
            
        return {'sample_idx':idx, 'images':images, 'lidar_points':lidar_points, 'instances':instances}
    
    ids = list(range(load_dict['total_number']))
    with futures.ThreadPoolExecutor(1) as executor:
        infos = executor.map(process_single_scene, ids)
    return list(infos)


def create_data_info(data_path, file_list, pkl_prefix='STCrowd'):
    # only deal with train split
    path = data_path +'anno/'
    all_files = [str(file)+'.json' for file in file_list]
    data_list = []
    for file in all_files:
        file_group_path = ''.join([path, file])
        with open(file_group_path, 'r') as load_f:
            load_dict = json.load(load_f)
        info = load_file(data_path, load_dict)    
        if info:
            data_list = data_list + info
    metainfo = {'categories': {'person': 0}, 'dataset': 'STCrowd', 'info_version': '1.0'}
    return {'metainfo': metainfo, 'data_list': data_list}


def main():
    args = options()
    path_root = args.path_root
    split = args.split
    split_file = args.path_root + args.split_file
    with open(split_file, 'r') as load_f:
        load_dict = json.load(load_f)
    info = create_data_info(data_path=path_root, file_list=load_dict[split])
    filename = f'{path_root}STCrowd_infos_{split}.pkl'
    print(f'dataset info {split} file is saved to {filename}')
    mmengine.dump(info, filename)


if __name__ == "__main__":
    main()