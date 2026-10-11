# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# (adapted for lightweight data-generation: kasal dependency replaced with pymeshlab + numpy)

import os, json, argparse
import numpy as np
import pymeshlab as ml


def get_all_ply_obj(dir):
    """Collect all .ply/.obj files under dir, excluding *_sym.ply visualization files."""
    filelist = []
    for home, dirs, files in os.walk(dir):
        for filename in files:
            if (filename.endswith('.ply') or filename.endswith('.obj')) and not filename.endswith('_sym.ply'):
                filelist.append(os.path.join(home, filename))
    return filelist


def load_json2dict(path):
    with open(path, 'r') as f:
        return json.load(f)


def write_dict2json(path, d):
    with open(path, 'w') as f:
        json.dump(d, f, indent=4)


def load_ply_model(path):
    """Load a PLY mesh and return its vertices (Nx3) and diameter (max pairwise distance)."""
    ms = ml.MeshSet()
    ms.load_new_mesh(path)
    vertices = ms.current_mesh().vertex_matrix().copy()
    # Diameter = max distance between any two vertices == max distance on the convex hull.
    ms.generate_convex_hull()
    hull = ms.current_mesh().vertex_matrix()
    diff = hull[:, None, :] - hull[None, :, :]
    diameter = float(np.sqrt((diff * diff).sum(axis=-1)).max())
    return {'vertices': vertices, 'diameter': diameter}


if __name__ == '__main__':

    # Specify the dataset path. The `dataset_path` directory must contain a `models` folder.
    # 指定数据集的路径。`dataset_path`目录下必须包含一个`models`文件夹。
    '''
    my_dataset
    |--- models
        |--- obj_000001.ply
    '''
    parser = argparse.ArgumentParser(description='Generate models_info.json from BOP-format PLY files.')
    parser.add_argument('dataset_path', type=str, help='Dataset root directory (must contain a models/ folder).')
    args = parser.parse_args()
    dataset_path = args.dataset_path

    # Retrieve all PLY files from the folder and its subfolders.
    # 获取该文件夹及其所有子文件夹中的所有PLY文件。
    models_path = get_all_ply_obj(dataset_path)

    # Iterate over all PLY files to compute the diameter, per-axis min/max of vertex coordinates,
    # and the bounding-box side lengths. Load the rotational-symmetry prior file,
    # then merge everything into `models_info.json`.
    # 遍历所有 PLY 文件，计算直径、顶点坐标各轴的最小/最大值，以及包围盒边长。
    # 加载旋转对称先验文件，最后合并所有信息生成 `models_info.json`。
    models_info = {}
    for model_path in models_path:
        ply_info = load_ply_model(model_path)
        verts = ply_info['vertices']
        vmin = verts.min(axis=0)
        vmax = verts.max(axis=0)
        model_info = {
            "diameter": ply_info['diameter'],
            "max_x": float(vmax[0]),
            "max_y": float(vmax[1]),
            "max_z": float(vmax[2]),
            "min_x": float(vmin[0]),
            "min_y": float(vmin[1]),
            "min_z": float(vmin[2]),
            "size_x": float(vmax[0] - vmin[0]),
            "size_y": float(vmax[1] - vmin[1]),
            "size_z": float(vmax[2] - vmin[2]),
        }
        symmetry_type_dict = None
        sym_type_file = os.path.join(os.path.dirname(model_path), os.path.basename(model_path).split('.')[0] + '_sym_type.json')
        if os.path.exists(sym_type_file):
            symmetry_type_dict = load_json2dict(sym_type_file)
        if symmetry_type_dict is not None:
            if 'symmetries_continuous' in symmetry_type_dict['current_obj_info']:
                model_info["symmetries_continuous"] = symmetry_type_dict['current_obj_info']["symmetries_continuous"]
            if 'symmetries_discrete' in symmetry_type_dict['current_obj_info']:
                model_info["symmetries_discrete"] = symmetry_type_dict['current_obj_info']["symmetries_discrete"]
        model_id = str(int(os.path.basename(model_path).split('.')[0].split('obj_')[1]))
        models_info[model_id] = model_info

    # Check whether the folder contains any objects; if so, generate `models_info.json`.
    # 判断该文件夹是否包含物体；若包含，则生成 `models_info.json`。
    if len(models_path) > 0:
        models_info_path = os.path.join(os.path.dirname(models_path[0]), 'models_info.json')
        write_dict2json(models_info_path, models_info)
        print('Wrote %d objects -> %s' % (len(models_info), models_info_path))
    pass
