import os
import os.path as osp
import torch
import numpy as np
import mmcv
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmdet3d.utils import replace_ceph_backend
from projects.mmdet3d_plugin.models.detectors import CmtDetector
import time
from mmengine.structures import InstanceData
from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import LiDARInstance3DBoxes

cfg = Config.fromfile('projects/configs/fusion/my_cmt_kitti_pre.py')
# cfg = Config.fromfile('../mmdetection3d/projects/BEVFusion/configs/my_bevfusion.py')
cfg.work_dir = osp.abspath('./work_dirs')
runner = Runner.from_cfg(cfg)
visualizer1 = Det3DLocalVisualizer()
visualizer2 = Det3DLocalVisualizer()
# visualizer3 = Det3DLocalVisualizer()

# runner.train_dataloader.dataset.dataset.pipeline.transforms.pop(3)
# runner.load_checkpoint('models/epoch_120.pth')
# runner.model.eval()
count = 0
for data_batch in runner.train_dataloader:
    if count == 0:
        break
    count -= 1
data_batch = runner.model.data_preprocessor(data_batch, training=False)
batch_inputs_dict = data_batch['inputs']
batch_data_samples = data_batch['data_samples']
imgs = batch_inputs_dict.get('imgs', None)
points = batch_inputs_dict.get('points', None)
img_metas = [item.metainfo for item in batch_data_samples]
# gt_bboxes_3d = [item.get('eval_ann_info')['gt_bboxes_3d'] for item in batch_data_samples]
# gt_labels_3d = [item.get('eval_ann_info')['gt_labels_3d'] for item in batch_data_samples]
gt_bboxes_3d = [item.get('gt_instances_3d')['bboxes_3d'] for item in batch_data_samples]
gt_labels_3d = [item.get('gt_instances_3d')['labels_3d'] for item in batch_data_samples]

point = points[0].cpu().numpy()
bboxes_3d = gt_bboxes_3d[0]
bbox_color = [(0, 255, 0)] * bboxes_3d.shape[0]
input_meta = {'lidar2img':img_metas[0]['lidar2img'][0]}
img = imgs[0][0].permute(1, 2, 0).cpu().numpy()
img = mmcv.imdenormalize(img, mean=np.array([103.530, 116.280, 123.675]), std=np.array([57.375, 57.120, 58.395]), to_bgr=True)

# if isinstance(data_batch, dict):
#     outputs = runner.model(**data_batch, mode='predict')
# elif isinstance(data_batch, (list, tuple)):
#     outputs = runner.model(**data_batch, mode='predict')
# else:
#     raise TypeError()
# runner.val_evaluator.process(data_samples=outputs, data_batch=data_batch)

# bboxes_3d_pre = outputs[0].get('pred_instances_3d').get('bboxes_3d')[:20]
# scores_3d_pre = outputs[0].get('pred_instances_3d').get('scores_3d')[:20]
# bbox_color_pre = [(0, 255, 0)] * bboxes_3d_pre.shape[0]
# print(scores_3d_pre)

visualizer1.set_points(point)
visualizer1.draw_bboxes_3d(bboxes_3d, bbox_color)

visualizer2.set_image(img)
visualizer2.draw_proj_bboxes_3d(bboxes_3d, input_meta)
# print(img_metas[0]['filename'])

# visualizer3.set_points(point)
# visualizer3.draw_bboxes_3d(bboxes_3d_pre, bbox_color_pre)

visualizer1.show()
visualizer2.show()
# visualizer3.show()