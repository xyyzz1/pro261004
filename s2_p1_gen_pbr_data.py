# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

'''
s2_p1_gen_pbr_data.py is used to generate PBR data.  
The original script is adapted from BlenderProc2.  
Project link: https://github.com/DLR-RM/BlenderProc

Usage:
    cd HCCEPose
    chmod +x s2_p1_gen_pbr_data.sh
    
    cc0textures:
    nohup ./s2_p1_gen_pbr_data.sh 0 42 xxx/xxx/cc0textures xxx/xxx/demo-bin-picking xxx/xxx/s2_p1_gen_pbr_data.py > s2_p1_gen_pbr_data.log 2>&1 &

    cc0textures-512:
    nohup ./s2_p1_gen_pbr_data.sh 0 42 xxx/xxx/cc0textures-512 xxx/xxx/demo-bin-picking xxx/xxx/s2_p1_gen_pbr_data.py > s2_p1_gen_pbr_data.log 2>&1 &

    
Arguments (example: s2_p1_gen_pbr_data.sh 0 42 ... ):
    Arg 1 (`GPU_ID`): GPU index. Set to 0 for the first GPU.
    Arg 2 (`SCENE_NUM`): Number of scenes; total images generated = 1000 * 42.
    Arg 3 (`cc0textures`): Path to the cc0textures material library.
    Arg 4 (`dataset_path`): Path to the dataset.
    Arg 4 (`s2_p1_gen_pbr_data`): Path to the s2_p1_gen_pbr_data.py script.
    
------------------------------------------------------    

s2_p1_gen_pbr_data.py 用于生成 PBR 数据，原始脚本改编自 BlenderProc2。  
项目链接: https://github.com/DLR-RM/BlenderProc

运行方法:
    cd HCCEPose
    chmod +x s2_p1_gen_pbr_data.sh
    
    cc0textures:
    nohup ./s2_p1_gen_pbr_data.sh 0 42 xxx/xxx/cc0textures xxx/xxx/demo-bin-picking xxx/xxx/s2_p1_gen_pbr_data.py > s2_p1_gen_pbr_data.log 2>&1 &

    cc0textures-512:
    nohup ./s2_p1_gen_pbr_data.sh 0 42 xxx/xxx/cc0textures-512 xxx/xxx/demo-bin-picking xxx/xxx/s2_p1_gen_pbr_data.py > s2_p1_gen_pbr_data.log 2>&1 &

参数说明 (以 s2_p1_gen_pbr_data.sh 0 42 ... 为例):
    参数 1 (`GPU_ID`): GPU 的编号。设置为 0 表示使用第一块显卡。
    参数 2 (`SCENE_NUM`): 场景数量，对应生成的图像数 = 1000 * 42
    参数 3 (`cc0textures`): cc0textures 材质库的路径。
    参数 3 (`dataset_path`): 数据集的路径。
    参数 3 (`s2_p1_gen_pbr_data`): s2_p1_gen_pbr_data.py的路径。
'''

import os
import json
import bpy
import argparse
import blenderproc as bproc
import numpy as np
from tqdm import tqdm
from mathutils import Matrix, Vector
from bpy_extras.object_utils import world_to_camera_view


# Lightweight JSON helpers, replacing the previous `kasal.utils.io_json` dependency.
# (kasal's top-level `__init__` pulls in a full GUI stack — cv2/open3d/polyscope/dash —
#  which is unnecessary here, so these two functions are inlined instead.)
def load_json2dict(path):
    with open(path, 'r') as f:
        return json.load(f)


def write_dict2json(path, d):
    with open(path, 'w') as f:
        json.dump(d, f, indent=4)


def objects_in_frame(objects, cam2world, img_w, img_h, margin):
    """Return True if all 8 corners of every object's bbox project strictly inside the frame.

    Uses Blender's own `world_to_camera_view` (which matches the rendered image exactly,
    including the camera's -Z forward / +Y up convention) by temporarily setting the scene
    camera's world matrix to the candidate pose. `margin` is the minimum distance in pixels
    of any corner to the image border.

    用 Blender 自带的 world_to_camera_view（与渲染图完全一致，含相机 -Z 前向/+Y 向上约定），
    把每个物体包围盒的 8 个角点投影到像素坐标，检查是否都落在画面内（留 margin 像素边距），
    用于保证“物体不超出画面边界”。
    """
    cam_ob = bpy.context.scene.camera
    cam_ob.matrix_world = Matrix(cam2world)
    scene = bpy.context.scene
    for obj in objects:
        for corner in obj.get_bound_box():  # (8, 3) world coords, metres
            v = world_to_camera_view(scene, cam_ob, Vector(corner))
            if v.z <= 0.0:  # behind the camera
                return False
            u = v.x * img_w
            vv = v.y * img_h
            if u < margin or u > img_w - margin or vv < margin or vv > img_h - margin:
                return False
    return True


if __name__ == '__main__':
    
    # Retrieve the GPU ID.
    # 获取 GPU 的编号。
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('gpu_id', type=int, help='')
    parser.add_argument('cc0textures', type=str, help='')
    parser.add_argument('--num_scenes', type=int, default=50, help='Number of scenes to render (default 50).')
    parser.add_argument('--frames', type=int, default=20, help='Camera frames per scene (default 20).')
    parser.add_argument('--device', type=str, default='CUDA', help='Render device: CUDA/OPTIX/CPU (default CUDA).')
    parser.add_argument('--min_objs', type=int, default=1, help='Minimum instances per image (default 1).')
    parser.add_argument('--max_objs', type=int, default=4, help='Maximum instances per image (default 4).')
    parser.add_argument('--cam_radius_min', type=float, default=0.20, help='Camera min radius [m] around the object cluster (default 0.20).')
    parser.add_argument('--cam_radius_max', type=float, default=0.30, help='Camera max radius [m] around the object cluster (default 0.30).')
    parser.add_argument('--margin', type=int, default=20, help='Min distance [px] of any object corner to the image border (default 20).')
    args = parser.parse_args()
    gpu_id = int(args.gpu_id)

    # Retrieve the folder path of the current dataset.
    # 获取当前数据集的文件夹路径。
    current_dir = os.path.abspath(os.getcwd())

    # Retrieve the name of the dataset.
    # 获取数据集的名称。
    dataset_name = os.path.basename(current_dir) 

    # BlenderProc requires the path to the parent directory of the dataset.
    # BlenderProc 需要传入数据集的父级目录路径。
    bop_parent_path = os.path.dirname(current_dir)

    # Load the 3D model information of the dataset.
    # 加载数据集的 3D 模型信息。
    models_info = load_json2dict(os.path.join(current_dir, 'models', 'models_info.json'))

    if not os.path.exists(os.path.join(current_dir, 'camera.json')):
        write_dict2json(os.path.join(current_dir, 'camera.json'), 
                            {
                            "cx": 325.2611083984375,
                            "cy": 242.04899588216654,
                            "depth_scale": 0.1,
                            "fx": 572.411363389757,
                            "fy": 573.5704328585578,
                            "height": 480,
                            "width": 640
                            }
                        )

    # Camera resolution, used by the in-frame visibility check.
    # 相机分辨率，用于“物体不超出画面”的可视性检查。
    camera_meta = load_json2dict(os.path.join(current_dir, 'camera.json'))
    img_w = int(camera_meta['width'])
    img_h = int(camera_meta['height'])

    # Retrieve the list of 3D model IDs from the dataset.
    # 获取数据集中 3D 模型的 ID 列表。
    models_ids = []
    for key in models_info:
        models_ids.append(int(key))
    models_ids = np.array(models_ids)

    # Print the parent path and name of the dataset.
    # 打印数据集的父级路径和名称。
    print('-*' * 10)
    print('-*' * 10)
    print('bop_parent_path', bop_parent_path)
    print('dataset_name', dataset_name)
    print('-*' * 10)
    print('-*' * 10)

    # Retrieve the path to the cc0textures assets.
    # 获取 cc0textures 的路径。
    cc_textures_path = args.cc0textures

    bop_dataset_path = os.path.join(bop_parent_path, dataset_name)
    num_scenes = args.num_scenes

    # Create the rendering scene.
    # 创建渲染场景。
    bproc.init()
    bproc.loader.load_bop_intrinsics(bop_dataset_path = bop_dataset_path)
    room_planes = [bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
                bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]),
                bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]),
                bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]),
                bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0])]
    for plane in room_planes:
        plane.enable_rigidbody(False, collision_shape='BOX', mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
    light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
    light_plane.set_name('light_plane')
    light_plane_material = bproc.material.create('light_material')
    light_point = bproc.types.Light()
    light_point.set_energy(200)

    # Load all texture images from the cc_textures directory.
    # 加载 cc_textures 目录中的所有纹理图。
    if os.path.basename(cc_textures_path) == 'cc0textures-512':
        cc_textures = bproc.loader.load_512_ccmaterials(cc_textures_path, use_all_materials=True)
    else:
        cc_textures = bproc.loader.load_ccmaterials(cc_textures_path, use_all_materials=True)

    
    def sample_pose_func(obj: bproc.types.MeshObject):
        # Tighter placement so 2~6 instances stay well inside the frame with the closer camera.
        # 更紧凑的放置范围，配合拉近的相机，使 2~6 个实例尽量完整落在画面内。
        min = np.random.uniform([-0.08, -0.08, 0.0], [-0.05, -0.05, 0.0])
        max = np.random.uniform([0.05, 0.05, 0.35], [0.08, 0.08, 0.5])
        obj.set_location(np.random.uniform(min, max))
        obj.set_rotation_euler(bproc.sampler.uniformSO3())
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.set_max_amount_of_samples(50)

    # Set the GPU ID.
    # 设置 GPU 的编号。
    if args.device.upper() == 'CPU':
        bproc.renderer.set_render_devices(use_only_cpu=True)
    else:
        bproc.renderer.set_render_devices(desired_gpu_device_type=args.device, desired_gpu_ids = [gpu_id])

    for i in tqdm(range(num_scenes)):
        
        rand_s = np.random.rand()

        # Bin-picking selection mode.
        # bin-picking 的挑选模式。

        # idx_l = np.random.choice(models_ids, size=2, replace=True)
        # obj_ids = []
        # for _ in range(15):
        #     obj_ids.append(int(idx_l[0]))
        #     obj_ids.append(int(idx_l[1]))

        # Multi-class object picking mode.
        # 多类别物体的挑选模式。

        # Place 2~6 instances per image (configurable via --min_objs / --max_objs).
        # 每张图放置 2~6 个实例（可通过 --min_objs / --max_objs 调整）。
        num_objs = int(np.random.randint(args.min_objs, args.max_objs + 1))
        idx_l = np.random.choice(models_ids, size=num_objs, replace=True)
        obj_ids = []
        for idx_i in idx_l:
            obj_ids.append(int(idx_i))
        
        # Load objects into BlenderProc.
        # 将物体加载到 BlenderProc 中。
        target_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = bop_dataset_path, 
                                                    mm2m = True,
                                                    obj_ids = obj_ids,
                                                    )
        
        # Set object materials and poses, then render 20 frames.
        # 设置物体的材质和位姿，并渲染 20 帧图像。
        
        for obj in (target_bop_objs):
            obj.set_shading_mode('auto')
            obj.hide(True)
        sampled_target_bop_objs = target_bop_objs
        for obj in (sampled_target_bop_objs):      
            mat = obj.get_materials()[0]     
            mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
            mat.set_principled_shader_value("Specular", np.random.uniform(0, 1.0))
            obj.enable_rigidbody(True, mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
            obj.hide(False)
        light_plane_material.make_emissive(emission_strength=np.random.uniform(3,6), 
                                        emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))  
        light_plane.replace_materials(light_plane_material)
        light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
        location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 1, radius_max = 1.5,
                                elevation_min = 5, elevation_max = 89)
        light_point.set_location(location)
        random_cc_texture = np.random.choice(cc_textures)
        for plane in room_planes:
            plane.replace_materials(random_cc_texture)

        bproc.object.sample_poses(objects_to_sample = sampled_target_bop_objs,
                                sample_pose_func = sample_pose_func, 
                                max_tries = 1000)
                
        bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                                        max_simulation_time=10,
                                                        check_object_interval=1,
                                                        substeps_per_frame = 20,
                                                        solver_iters=25)

        bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_target_bop_objs)

        cam_poses = 0
        attempts = 0
        max_attempts = 200
        while cam_poses < args.frames and attempts < max_attempts:
            attempts += 1
            # Sample the look-at point (object cluster centroid) first, then place the
            # camera on a shell centered on THAT point rather than on the world origin.
            # This keeps the camera-object distance — and hence the instance size —
            # consistent no matter where physics finally settles the pile, which is
            # what prevents the closer camera from clipping objects near the pile edges.
            # 先在物体簇中采样注视点(感兴趣点)，再把相机放到以该点为中心的球壳上，
            # 而不是以世界原点为中心。这样无论物理仿真最终把物体堆落在哪里，
            # 相机到物体的距离(即实例尺寸)都保持一致，避免近距离相机裁掉堆边缘的物体。
            poi = bproc.object.compute_poi(np.random.choice(sampled_target_bop_objs, size=int(round(0.6 * len(obj_ids))), replace=False))
            location = bproc.sampler.shell(center = poi,
                                    radius_min = args.cam_radius_min,
                                    radius_max = args.cam_radius_max,
                                    elevation_min = 5,
                                    elevation_max = 89)
            rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-3.14159, 3.14159))
            cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
            # Accept the pose only if the camera is not inside/too close to an object AND
            # every object corner projects inside the frame (no object exceeds the border).
            # 只有当相机不穿模且所有物体角点都落在画面内（不超出边界）时才接受该位姿。
            if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.05}, bop_bvh_tree) and \
               objects_in_frame(sampled_target_bop_objs, cam2world_matrix, img_w, img_h, args.margin):
                bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
                cam_poses += 1

        if cam_poses == 0:
            # No valid camera pose found for this scene; skip writing to avoid an empty chunk.
            # 未找到合法相机位姿，跳过写入以避免产生空 chunk。
            for obj in (sampled_target_bop_objs):
                obj.disable_rigidbody()
                obj.hide(True)
            continue

        data = bproc.renderer.render()
        bproc.writer.write_bop(bop_parent_path,
                            target_objects = sampled_target_bop_objs,
                            dataset = dataset_name,
                            depth_scale = 0.1,
                            depths = data["depth"],
                            colors = data["colors"],
                            color_file_format = "JPEG",
                            ignore_dist_thres = 10)
        
        for obj in (sampled_target_bop_objs):    
            obj.disable_rigidbody()  
            obj.hide(True)
    pass