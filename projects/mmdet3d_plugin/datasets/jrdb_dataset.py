from typing import Callable, List, Union
import os
from os import path as osp

import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.datasets import Det3DDataset


@DATASETS.register_module()
class JRDBDataset(Det3DDataset):
    r"""KITTI Dataset.

    This class serves as the API for experiments on the `KITTI Dataset
    <http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d>`_.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (List[dict]): Pipeline used for data processing.
            Defaults to [].
        modality (dict): Modality to specify the sensor data used as input.
            Defaults to dict(use_lidar=True).
        default_cam_key (str): The default camera name adopted.
            Defaults to 'CAM2'.
        load_type (str): Type of loading mode. Defaults to 'frame_based'.

            - 'frame_based': Load all of the instances in the frame.
            - 'mv_image_based': Load all of the instances in the frame and need
              to convert to the FOV-based data type to support image-based
              detector.
            - 'fov_image_based': Only load the instances inside the default
              cam, and need to convert to the FOV-based data type to support
              image-based detector.
        box_type_3d (str): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes:

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool): Whether to filter the data with empty GT.
            If it's set to be True, the example with empty annotations after
            data pipeline will be dropped and a random example will be chosen
            in `__getitem__`. Defaults to True.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (List[float]): The range of point cloud used to filter
            invalid predicted boxes.
            Defaults to [0, -40, -3, 70.4, 40, 0.0].
    """
    # TODO: use full classes of kitti
    METAINFO = {
        'classes': ('Pedestrian'),
        'palette': [(106, 0, 228)]
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True),
                 default_cam_key: str = None,
                 load_type: str = 'frame_based',
                 box_type_3d: str = 'LiDAR',
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 pcd_limit_range: List[float] = [0, -40, -3, 70.4, 40, 0.0],
                 **kwargs) -> None:

        self.pcd_limit_range = pcd_limit_range
        assert load_type in ('frame_based', 'mv_image_based',
                             'fov_image_based')
        self.load_type = load_type
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            modality=modality,
            default_cam_key=default_cam_key,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)
        assert self.modality is not None
        assert box_type_3d.lower() in ('lidar', 'camera')

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        The only difference with it in `Det3DDataset`
        is the specific process for `plane`.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        if self.modality['use_lidar']:
            if 'plane' in info:
                # convert ground plane to velodyne coordinates
                plane = np.array(info['plane'])
                lidar2cam = np.array(
                    info['images']['left']['lidar2cam'], dtype=np.float32)
                reverse = np.linalg.inv(lidar2cam)

                (plane_norm_cam, plane_off_cam) = (plane[:3],
                                                   -plane[:3] * plane[3])
                plane_norm_lidar = \
                    (reverse[:3, :3] @ plane_norm_cam[:, None])[:, 0]
                plane_off_lidar = (
                    reverse[:3, :3] @ plane_off_cam[:, None][:, 0] +
                    reverse[:3, 3])
                plane_lidar = np.zeros_like(plane_norm_lidar, shape=(4, ))
                plane_lidar[:3] = plane_norm_lidar
                plane_lidar[3] = -plane_norm_lidar.T @ plane_off_lidar
            else:
                plane_lidar = None

            info['plane'] = plane_lidar

        if self.load_type == 'fov_image_based' and self.load_eval_anns:
            info['instances'] = info['cam_instances'][self.default_cam_key]

        if self.modality['use_lidar']:
            info['lidar_points']['lidar_path_upper'] = \
                osp.join(
                    self.data_prefix.get('pts', ''),
                    info['lidar_points']['lidar_path_upper'])
            info['lidar_points']['lidar_path_lower'] = \
                osp.join(
                    self.data_prefix.get('pts', ''),
                    info['lidar_points']['lidar_path_lower'])
            info['num_pts_feats'] = info['lidar_points']['num_pts_feats']

        if self.modality['use_camera']:
            for cam_id, img_info in info['images'].items():
                if 'img_path' in img_info:
                    if cam_id in self.data_prefix:
                        cam_prefix = self.data_prefix[cam_id]
                    else:
                        cam_prefix = self.data_prefix.get('img', '')
                    img_info['img_path'] = osp.join(cam_prefix, img_info['img_path'])

            if self.default_cam_key is not None:
                info['img_path'] = info['images'][
                    self.default_cam_key]['img_path']
                if 'lidar2cam' in info['images'][self.default_cam_key]:
                    info['lidar2cam'] = np.array(
                        info['images'][self.default_cam_key]['lidar2cam'])
                if 'cam2img' in info['images'][self.default_cam_key]:
                    info['cam2img'] = np.array(
                        info['images'][self.default_cam_key]['cam2img'])
                if 'lidar2img' in info['images'][self.default_cam_key]:
                    info['lidar2img'] = np.array(
                        info['images'][self.default_cam_key]['lidar2img'])
                else:
                    info['lidar2img'] = info['cam2img'] @ info['lidar2cam']

        if not self.test_mode:
            # used in training
            info['ann_info'] = self.parse_ann_info(info)
        if self.test_mode and self.load_eval_anns:
            info['eval_ann_info'] = self.parse_ann_info(info)

        return info

    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                  3D ground truth bboxes.
                - bbox_labels_3d (np.ndarray): Labels of ground truths.
                - gt_bboxes (np.ndarray): 2D ground truth bboxes.
                - gt_labels (np.ndarray): Labels of ground truths.
                - difficulty (int): Difficulty defined by KITTI.
                  0, 1, 2 represent xxxxx respectively.
        """
        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            ann_info = dict()
            # empty instance
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

            if self.load_type in ['fov_image_based', 'mv_image_based']:
                ann_info['gt_bboxes'] = np.zeros((0, 4), dtype=np.float32)
                ann_info['gt_bboxes_labels'] = np.array(0, dtype=np.int64)
                ann_info['centers_2d'] = np.zeros((0, 2), dtype=np.float32)
                ann_info['depths'] = np.zeros((0), dtype=np.float32)

        ann_info = self._remove_dontcare(ann_info)
        ann_info['gt_bboxes_3d'] = LiDARInstance3DBoxes(ann_info['gt_bboxes_3d'])
        return ann_info