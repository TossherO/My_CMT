import torch
import numpy as np
import cv2
import mmcv
import mmengine
import time
import open3d as o3d
from open3d import geometry
import os
from mmengine.structures import InstanceData
from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import LiDARInstance3DBoxes, CameraInstance3DBoxes

visualizer1 = Det3DLocalVisualizer()
visualizer2 = Det3DLocalVisualizer()

infos = mmengine.load('./data/kitti/kitti_infos_trainval2.pkl')

idx = 0
info = infos['data_list'][idx]

cam2img = []
lidar2cam = []
lidar2img = []
xyz2zxy = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
for cam in ['CAM2']:
    cam2img.append(np.array(info['images'][cam]['cam2img']))
    lidar2cam.append(np.array(info['images'][cam]['lidar2cam']))
    lidar2img.append(np.array(info['images'][cam]['cam2img']) @ np.array(info['images'][cam]['lidar2cam']))

cam = 'CAM2'
img_path = './data/kitti/' + info['images'][cam]['img_path']
points_path = './data/kitti/' + info['lidar_points']['lidar_path']
input_meta = {'lidar2img': np.array(info['images'][cam]['lidar2img'])}

img = cv2.imread(img_path)
img = img.astype(np.float32)
points = np.fromfile(points_path, dtype=np.float32).reshape([-1, 4])

bbox_3d = np.array([item['bbox_3d'] for item in info['instances']])
bboxes_3d = LiDARInstance3DBoxes([instance['bbox_3d'] for instance in info['instances']])
bbox_color = [(0, 255, 0)] * bboxes_3d.shape[0]

visualizer1.set_points(points)
visualizer1.draw_bboxes_3d(bboxes_3d, bbox_color)

# 将相机视角绘制在3D点云坐标系中
# 获取图像的尺寸
img_h, img_w = img.shape[:2]
# 生成的图像框
img_bbox = np.array([[0, 0, 1], [img_w/2, 0, 1], [img_w, 0, 1], [img_w, img_h/2, 1], [img_w, img_h, 1], [img_w/2, img_h, 1], [0, img_h, 1], [0, img_h/2, 1]])
# 生成不同深度的图像框
img_bboxes = np.concatenate([img_bbox * d for d in range(1, 6, 1)])
img_bboxes = np.concatenate([img_bboxes, np.ones((img_bboxes.shape[0], 1))], axis=1)
img2lidar_bboxes = []
for i in range(len(lidar2img)):
    img2lidar_bboxes.append(np.dot(img_bboxes, np.linalg.inv(lidar2img[i]).T))
img2lidar_bboxes = np.concatenate(img2lidar_bboxes, axis=0)[:, :3].reshape(-1, 8, 3)
line_colors = np.array([[[1, 0, 0]] * 8, [[0, 1, 0]] * 8, [[0, 0, 1]] * 8, [[1, 1, 0]] * 8, [[0, 1, 1]] * 8, [[1, 0, 1]] * 8])
print(img2lidar_bboxes.shape)
# 将图像框绘制到点云坐标系中
for i in range(img2lidar_bboxes.shape[0]):
    lines = geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(img2lidar_bboxes[i])
    lines.lines = o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 0]])
    lines.colors = o3d.utility.Vector3dVector(line_colors[i//5])
    visualizer1.o3d_vis.add_geometry(lines)

visualizer2.set_image(img)
visualizer2.draw_proj_bboxes_3d(bboxes_3d, input_meta)

visualizer1.show()
visualizer2.show()