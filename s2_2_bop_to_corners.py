# Author: (adapted for HccePose(BF) s2 -> corner dataset)
# s2_2_bop_to_corners.py
#
# Read the BOP-format output produced by s2 (s2_p1_gen_pbr_data.py) and convert it
# into a flat, training-friendly dataset:
#
#   dataset/
#   ├── images/          # rgb images, renamed sequentially 000001.jpg, 000002.jpg, ...
#   ├── annotations.json # per-image object instances + the 8 projected 3D-bbox corners
#   └── bop_raw/         # the original BOP data (models/ + train_pbr/ + camera.json ...)
#
# The "8 corners" are the 8 vertices of each object's axis-aligned 3D bounding box
# (from models_info.json min_*/max_*), projected into the image plane via the object's
# ground-truth pose (cam_R_m2c / cam_t_m2c) and the camera intrinsics (cam_K).
#
# Corner order (z slowest, then y, then x, each min then max):
#   [0] (min_x, min_y, min_z)  [1] (max_x, min_y, min_z)
#   [2] (min_x, max_y, min_z)  [3] (max_x, max_y, min_z)
#   [4] (min_x, min_y, max_z)  [5] (max_x, min_y, max_z)
#   [6] (min_x, max_y, max_z)  [7] (max_x, max_y, max_z)
#
# Usage:
#   python s2_2_bop_to_corners.py /path/to/my_dataset --out_dir /path/to/dataset
#
# 本脚本读取 s2 (s2_p1_gen_pbr_data.py) 生成的 BOP 格式数据，转换成平铺的训练友好数据集：
#   images/ 按顺序重命名 rgb 图片；annotations.json 记录每张图每个实例的 8 个 3D 包围盒角点
#   在图像上的投影坐标；bop_raw/ 保留原始 BOP 数据。
# "8 个角点" 是物体轴对齐 3D 包围盒（models_info.json 的 min_*/max_*）的 8 个顶点，
#   经 GT 位姿 (cam_R_m2c / cam_t_m2c) 和相机内参 (cam_K) 投影到图像平面得到。
# 角点顺序：z 最慢、y 次之、x 最快（各自 min→max），见上方英文注释。

import os
import json
import shutil
import argparse
import numpy as np


def load_json(path, keys_to_int=False):
    with open(path, 'r') as f:
        d = json.load(f)
    if keys_to_int:
        return {int(k): v for k, v in d.items()}
    return d


def build_corners_3d(min_xyz, max_xyz):
    """Build the 8 corners of an axis-aligned bbox in the fixed documented order."""
    xs = [float(min_xyz[0]), float(max_xyz[0])]
    ys = [float(min_xyz[1]), float(max_xyz[1])]
    zs = [float(min_xyz[2]), float(max_xyz[2])]
    corners = []
    for z in zs:
        for y in ys:
            for x in xs:
                corners.append([x, y, z])
    return np.array(corners, dtype=np.float64)  # (8, 3)


def project_pts(pts_3d, R, t, K):
    """Project 3D points (N,3) with pose R(3,3), t(3,) and intrinsics K(3,3) -> (N,2) pixels.

    pts_3d and t must be in the same unit (BOP stores both in mm); the unit cancels in
    the perspective division so the result is in pixels regardless.
    """
    # p_cam = R @ p_model + t  ->  row vectors: pts @ R.T + t
    p_cam = pts_3d @ R.T + t  # (N, 3)
    z = p_cam[:, 2]
    u = K[0, 0] * p_cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * p_cam[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=1)  # (N, 2)


def reshape_R(r_flat):
    return np.array(r_flat, dtype=np.float64).reshape(3, 3)


def reshape_K(k_flat):
    return np.array(k_flat, dtype=np.float64).reshape(3, 3)


def round_list(x, nd=4):
    return [round(float(v), nd) for v in x]


def round_mat(m, nd=6):
    return [[round(float(v), nd) for v in row] for row in m]


def find_rgb_file(scene_dir, frame_id):
    """Locate the rgb image for a frame, supporting both the subdir layout used by the
    HccePose fork (rgb/000000.jpg) and the standard BOP layout (rgb_000000.png)."""
    fid = '%06d' % frame_id
    candidates = [
        os.path.join(scene_dir, 'rgb', fid + '.jpg'),
        os.path.join(scene_dir, 'rgb', fid + '.png'),
        os.path.join(scene_dir, 'rgb_' + fid + '.jpg'),
        os.path.join(scene_dir, 'rgb_' + fid + '.png'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert s2 BOP output into images/ + annotations.json (8 projected corners) + bop_raw/.')
    parser.add_argument('dataset_path', type=str, help='Path to the BOP dataset root (contains models/, models_info.json, camera.json, train_pbr/).')
    parser.add_argument('--out_dir', type=str, default=None, help='Output directory (default: <dataset_path>/../<name>_corners).')
    parser.add_argument('--no_copy_raw', action='store_true', help='Do not copy the raw BOP data into bop_raw/.')
    args = parser.parse_args()

    ds_path = os.path.abspath(args.dataset_path)
    if args.out_dir:
        out_dir = os.path.abspath(args.out_dir)
    else:
        out_dir = os.path.join(os.path.dirname(ds_path), os.path.basename(ds_path) + '_corners')

    models_info = load_json(os.path.join(ds_path, 'models', 'models_info.json'))

    # Global object info: 8 3D corners per object (mm).
    objects_meta = {}
    for obj_id_str, info in models_info.items():
        obj_id = int(obj_id_str)
        min_xyz = [info['min_x'], info['min_y'], info['min_z']]
        max_xyz = [info['max_x'], info['max_y'], info['max_z']]
        corners3d = build_corners_3d(min_xyz, max_xyz)
        objects_meta[obj_id] = {
            'name': 'obj_%06d' % obj_id,
            'diameter': float(info['diameter']),
            'corners_3d_mm': round_mat(corners3d, 4),
        }

    # Camera intrinsics: prefer camera.json for the canonical K; scene_camera.json is
    # used per-frame below (they should match for a fixed camera).
    cam_file = os.path.join(ds_path, 'camera.json')
    camera_meta = {'width': 640, 'height': 480}
    if os.path.exists(cam_file):
        cam = load_json(cam_file)
        camera_meta = {
            'width': int(cam.get('width', 640)),
            'height': int(cam.get('height', 480)),
            'fx': float(cam.get('fx', 0.0)),
            'fy': float(cam.get('fy', 0.0)),
            'cx': float(cam.get('cx', 0.0)),
            'cy': float(cam.get('cy', 0.0)),
        }

    images_dir = os.path.join(out_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    annotations = {
        'meta': {
            'description': 's2 corner dataset: 8 projected 3D-bbox corners per object instance.',
            'corner_order': ['(min_x,min_y,min_z)', '(max_x,min_y,min_z)',
                             '(min_x,max_y,min_z)', '(max_x,max_y,min_z)',
                             '(min_x,min_y,max_z)', '(max_x,min_y,max_z)',
                             '(min_x,max_y,max_z)', '(max_x,max_y,max_z)'],
            'units': {'model': 'mm', 'cam_t_m2c': 'mm', 'corners_2d': 'pixels'},
        },
        'camera': camera_meta,
        'objects': objects_meta,
        'images': [],
    }

    train_pbr = os.path.join(ds_path, 'train_pbr')
    if not os.path.isdir(train_pbr):
        raise FileNotFoundError('No train_pbr/ directory found under %s' % ds_path)

    scene_dirs = sorted(
        [d for d in os.listdir(train_pbr) if os.path.isdir(os.path.join(train_pbr, d))],
        key=lambda x: int(x),
    )

    global_id = 0
    total_instances = 0

    for scene_name in scene_dirs:
        scene_dir = os.path.join(train_pbr, scene_name)
        gt_path = os.path.join(scene_dir, 'scene_gt.json')
        cam_path = os.path.join(scene_dir, 'scene_camera.json')
        info_path = os.path.join(scene_dir, 'scene_gt_info.json')
        if not (os.path.exists(gt_path) and os.path.exists(cam_path)):
            continue

        scene_gt = load_json(gt_path, keys_to_int=True)
        scene_cam = load_json(cam_path, keys_to_int=True)
        scene_info = load_json(info_path, keys_to_int=True) if os.path.exists(info_path) else {}

        for frame_id in sorted(scene_gt.keys()):
            global_id += 1
            frame_objs = scene_gt[frame_id]
            K = reshape_K(scene_cam[frame_id]['cam_K'])

            objects_list = []
            for i, gt_obj in enumerate(frame_objs):
                obj_id = int(gt_obj['obj_id'])
                R = reshape_R(gt_obj['cam_R_m2c'])
                t = np.array(gt_obj['cam_t_m2c'], dtype=np.float64)  # mm
                corners_3d = np.array(objects_meta[obj_id]['corners_3d_mm'], dtype=np.float64)
                corners_2d = project_pts(corners_3d, R, t, K)  # (8, 2) px

                entry = {
                    'obj_id': obj_id,
                    'cam_R_m2c': round_mat(R, 6),
                    'cam_t_m2c': round_list(t, 6),
                    'corners_2d': [[round(float(u), 2), round(float(v), 2)]
                                   for u, v in corners_2d],
                }
                if i < len(scene_info.get(frame_id, [])):
                    entry['bbox_visib'] = [int(v) for v in scene_info[frame_id][i].get('bbox_visib', [-1, -1, -1, -1])]
                    entry['visib_fract'] = float(scene_info[frame_id][i].get('visib_fract', 0.0))
                objects_list.append(entry)
                total_instances += 1

            # Copy and rename the rgb image.
            src_rgb = find_rgb_file(scene_dir, frame_id)
            dst_name = '%06d.jpg' % global_id
            dst_rgb = os.path.join(images_dir, dst_name)
            if src_rgb is not None and os.path.exists(src_rgb):
                shutil.copy2(src_rgb, dst_rgb)
            else:
                print('WARNING: no rgb image for scene %s frame %d' % (scene_name, frame_id))

            annotations['images'].append({
                'id': global_id,
                'file': os.path.join('images', dst_name),
                'scene_id': int(scene_name),
                'frame_id': frame_id,
                'cam_K': round_mat(K, 6),
                'objects': objects_list,
            })

    annotations['meta']['num_images'] = global_id
    annotations['meta']['num_instances'] = total_instances

    with open(os.path.join(out_dir, 'annotations.json'), 'w') as f:
        json.dump(annotations, f, indent=2)

    # Preserve the raw BOP data for reference.
    if not args.no_copy_raw:
        bop_raw = os.path.join(out_dir, 'bop_raw')
        os.makedirs(bop_raw, exist_ok=True)
        shutil.copytree(os.path.join(ds_path, 'models'), os.path.join(bop_raw, 'models'), dirs_exist_ok=True)
        shutil.copytree(train_pbr, os.path.join(bop_raw, 'train_pbr'), dirs_exist_ok=True)
        for name in ('models_info.json', 'camera.json'):
            src = os.path.join(ds_path, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(bop_raw, name))

    print('Done: %d images, %d instances -> %s' % (global_id, total_instances, out_dir))
