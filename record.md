# pro261004 环境搭建与 s1 / s2 跑通记录

目标：创建全新 conda 环境 `pro261004`，安装 HccePose(BF) 项目 **s1 / s2 数据生成** 相关依赖，跑通 **s1**（物体预处理）与 **s2**（BlenderProc PBR 渲染），并记录所有命令。

---

## 0. 最终结论

| 阶段 | 状态 | 产物 |
|------|------|------|
| 环境创建 | ✅ | `pro261004` = CPython 3.10.22 + pip 24.2 |
| s1 依赖 | ✅ | numpy / pymeshlab / tqdm 等 |
| **s1 跑通** | ✅ | `my_dataset/models/obj_000001.ply` + `models_info.json` |
| s2 依赖 | ✅ | bpy 3.6.0 + blenderproc 运行栈 + pyrender |
| **s2 跑通** | ✅ | `my_dataset/train_pbr/000000/`（rgb/depth/mask/… + 全部 GT 标注） |

---

## 1. 创建 conda 环境

```bash
# 注意：不要直接 `conda create -n pro261004 python=3.10`，
# conda-forge 源下会解析成 GraalPy（JVM 版 Python），必须显式指定 defaults 源。
conda create -n pro261004 -c defaults python=3.10 pip -y
conda activate pro261004
python --version    # Python 3.10.22
```

> 首次 `conda create -n pro261004 python=3.10` 得到 GraalPy，已删除后重装为 CPython。

## 2. 降级 pip（修复 pip 26.2.1 bug）

```bash
conda install -n pro261004 -c defaults "pip<25" -y
pip --version    # pip 24.2
```

## 3. 安装依赖（逐包安装，清华 PyPI 镜像）

> pip 24.2 一次装多包会触发 tree 解析崩溃，所以每个包单独一条 `pip install`；
> 大包加 `--no-cache-dir` 避免镜像缓存损坏。全程用 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。

```bash
# ---- s1 依赖 ----
pip install numpy==1.26.4
pip install pymeshlab==2023.12.post1
pip install kasal-6d==0.1.3 --no-deps        # 仅数据处理能力，不装 GUI 依赖链
pip install tqdm

# ---- s2 依赖：bpy（Blender 3.6.0 Python 模块）----
# README 里 `pip install bpy==3.6.0 --extra-index-url https://download.blender.org/pypi/` 已失效
# （该 pypi 索引下 /simple/bpy/ 已 404；PyPI 只剩 bpy 4.2+ 且全是 cp311）。
# 但官方 wheel 仍托管在扁平目录 /pypi/bpy/ 下，直接给 wheel 直链即可：
pip install "https://download.blender.org/pypi/bpy/bpy-3.6.0-cp310-cp310-manylinux_2_28_x86_64.whl"
#   大小 371MB，约 2MB/s，3~5 分钟；顺带装上 cython/zstandard。

# ---- s2 依赖：blenderproc 运行栈（import blenderproc 时被强制链式导入）----
pip install trimesh==4.2.2
pip install Pillow==11.3.0
pip install opencv-python==4.9.0.80
pip install opencv-contrib-python==4.9.0.80
pip install h5py==3.15.1
pip install scipy==1.15.3
pip install rich==14.2.0
pip install pypng
pip install imageio==2.37.2
pip install scikit-learn
pip install scikit-image
pip install matplotlib==3.10.7
pip install pytz

# ---- s2 依赖：BOP 写入器 GT mask/info 计算用 pyrender ----
pip install pyrender    # 自带 PyOpenGL / pyglet / freetype-py

# ---- s2 依赖：下载 cc0textures 材质库用 ----
pip install huggingface_hub==0.36.0
```

> `mathutils`/`bmesh` 无需单独安装：`import bpy` 后自动注册。

## 4. 下载材质库 cc0textures-512（585MB）

```bash
# 用 huggingface_hub 下载轻量版材质库（完整版 cc0textures 约 44GB，不必下）
python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='SEU-WYL/HccePose', repo_type='dataset', \
filename='cc0textures-512.zip', local_dir='/home/xyyzz1/pro261004')"

# 解压（zip 顶层为 cc0textures-512/，含 cc0textures/ 下 1654 套 *_color/_normal/_roughness.jpg）
python -c "import zipfile; zipfile.ZipFile('/home/xyyzz1/pro261004/cc0textures-512.zip').extractall('/home/xyyzz1/pro261004')"
```

## 5. 代码改动（脱离 kasal 依赖 + 冒烟测试参数 + 单位缩放）

1. **`s1_p1_obj_rename_center.py`**：加 `argparse`（`input_ply` / `--obj_id` / `--out_dir` / `--scale`）。
2. **`s1_p3_obj_infos.py`**：重写，去掉 kasal，改用 `numpy + pymeshlab`。
3. **`s2_p1_gen_pbr_data.py`**：内联 `load_json2dict/write_dict2json`（去 kasal）；新增 `--num_scenes`（默认 50）、`--frames`（默认 20）、`--device`（默认 CUDA）、`--min_objs`/`--max_objs`（每图实例数，默认 1/4）、`--cam_radius_min`/`--cam_radius_max`（相机半径，默认 0.20/0.30 m）、`--margin`（角点距画面边距，默认 20px）。
4. **单位修复**：模型为 cm 单位，BOP 约定 mm，故 s1 用 `--scale 10` 重生成。
5. **每图 1~4 实例 + 相机随机远近/角度 + 物体不超出边界**：物体数 `randint(min_objs, max_objs+1)`；相机球壳以**物体簇注视点(POI)** 为圆心，`elevation 5°~89°`、`azimuth 0~360°` 随机，半径 `--cam_radius_min/max` 随机；落点收紧到 ±[0.05,0.08]、障碍检查 `min 0.05`；新增 `objects_in_frame()` 用 Blender 自带 `world_to_camera_view` 投影每个物体 8 角点，**拒绝任何物体超出画面边界的相机位姿**（留 `--margin` 边距），超 200 次采样未命中则跳过该场景。

## 6. 跑通 s1

```bash
cd /home/xyyzz1/pro261004
conda activate pro261004

# ① 居中 + 重命名 + cm->mm（×10）缩放
python s1_p1_obj_rename_center.py /home/xyyzz1/pro261004/osmo_action4.ply \
    --obj_id 1 --scale 10 --out_dir /home/xyyzz1/pro261004/my_dataset/models

# ② 生成 models_info.json
python s1_p3_obj_infos.py /home/xyyzz1/pro261004/my_dataset
```

结果（mm）：`size_x=71.36, size_y=32.80, size_z=45.10, diameter=81.83`（与 Osmo Action 4 实尺一致）。

## 7. 跑通 s2（冒烟测试：CPU 渲染 2 场景 × 5 帧）

```bash
cd /home/xyyzz1/pro261004/my_dataset
conda activate pro261004

# blenderproc 是项目自带 fork，需把外层目录加入 PYTHONPATH
PYTHONPATH=/home/xyyzz1/pro261004/blenderproc \
python /home/xyyzz1/pro261004/s2_p1_gen_pbr_data.py \
    0 /home/xyyzz1/pro261004/cc0textures-512 \
    --num_scenes 2 --frames 5 --device CPU
```

完整流程：`bproc.init` → 读 `camera.json`（自动生成）→ 建房间/灯光 → 载入 1654 套材质 →
每场景载入 2~6 个物体 → 采样位姿 → 物理仿真（3s）→ 渲染 5 帧 → 写 BOP（rgb/depth/mask/mask_visib
+ scene_gt / scene_gt_info / scene_gt_coco / scene_camera）。

验证结果（每图 2~6 实例 + 放大）：每帧 5 实例，bbox 均值 162×153px（原始 ~51px 的 **3.2 倍**），
明显出界 0/50。

> ⚠️ **场景/chunk 语义**：BOP 里一个 `scene` = 一个 chunk 目录，`frames_per_chunk=1000`（默认）。
> 在**单进程内** `--num_scenes N` 只是循环 N 次并把帧追加到同一个 `000000/`（帧数到 1000 才自动开新 chunk），
> 所以 N>1 时会出现 `duplicate keys` 告警、且所有帧都落在 `000000/`。**正式大规模生成请用
> `s2_p1_gen_pbr_data.sh` 循环**（每个进程 1 个场景，逐进程追加，自动按 1000 帧切分场景）。

产物 `my_dataset/train_pbr/000000/`：rgb×10、depth×10、mask×100、mask_visib×100、4 个 json。

**正式大规模生成**（原脚本 42 场景 × 1000 帧/场景）用 shell 循环：

```bash
chmod +x s2_p1_gen_pbr_data.sh
# 参数：GPU_ID 场景数 cc0textures路径 数据集路径 脚本路径
nohup ./s2_p1_gen_pbr_data.sh 0 42 /home/xyyzz1/pro261004/cc0textures-512 \
    /home/xyyzz1/pro261004/my_dataset /home/xyyzz1/pro261004/s2_p1_gen_pbr_data.py \
    > s2_p1_gen_pbr_data.log 2>&1 &
```

## 8. s2_2：BOP → 角点数据集（`s2_2_bop_to_corners.py`）

读取 s2 生成的 BOP 数据，输出训练友好数据集：

```
dataset/
├── images/           # rgb 顺序重命名 000001.jpg、000002.jpg ...
├── annotations.json  # 每图每实例的 8 个 3D 包围盒角点投影坐标
└── bop_raw/          # 原始 BOP 数据（models/ + train_pbr/ + camera.json + models_info.json）
```

```bash
cd /home/xyyzz1/pro261004
conda activate pro261004
python s2_2_bop_to_corners.py /home/xyyzz1/pro261004/my_dataset \
    --out_dir /home/xyyzz1/pro261004/my_dataset_corners
# 跳过 bop_raw 拷贝：加 --no_copy_raw
```

**关键约定**：
- 8 角点 = `models_info.json` 的 min_*/max_* 组成的轴对齐包围盒 8 顶点，经 `cam_R_m2c`/`cam_t_m2c`
  和 `cam_K` 投影到像素坐标。角点顺序（z 最慢、y 次之、x 最快，各自 min→max）：
  `(min_x,min_y,min_z), (max_x,min_y,min_z), (min_x,max_y,min_z), (max_x,max_y,min_z),
   (min_x,min_y,max_z), (max_x,min_y,max_z), (min_x,max_y,max_z), (max_x,max_y,max_z)`。
- **单位均为 mm**：BOP 的 `cam_t_m2c` 与模型角点都是 mm（本项目 fork 输出即 mm，非 m），
  投影时 `x/z` 单位相消，结果为像素。
- `annotations.json` 结构：`meta`（角点顺序/单位）、`camera`（内参）、`objects`（每类 8 个 3D 角点 mm）、
  `images[]`（每图 `id`/`file`/`scene_id`/`frame_id`/`cam_K` + `objects[]` 含
  `obj_id`/`cam_R_m2c`/`cam_t_m2c`/`corners_2d`/`bbox_visib`/`visib_fract`）。
- 图片读取兼容两种布局：本 fork 的 `rgb/000000.jpg` 子目录，及标准 BOP 的 `rgb_000000.png`。
- 验证（10 图 50 实例）：角点投影包围盒 u[244,407] v[164,324] 完整包住 bbox_visib，
  中心 (324,240) 与 bbox 中心 (328,241) 一致。

## 9. 正式实验数据生成（100 张图）

每张图：相机角度（elevation 5°~89°、azimuth 0~360°）与远近（半径 0.20~0.30 m）随机、
物体数 1~4、姿态/位置/光线/背景均随机，且保证物体不超出画面边界。**每图 = 1 个独立场景 × 1 帧**。

```bash
cd /home/xyyzz1/pro261004/my_dataset
conda activate pro261004
rm -rf train_pbr   # 清掉旧数据，避免 append 残留

# ① 生成 100 张 BOP 图（100 场景 × 1 帧，约 16 分钟，CPU）
PYTHONPATH=/home/xyyzz1/pro261004/blenderproc \
python /home/xyyzz1/pro261004/s2_p1_gen_pbr_data.py \
    0 /home/xyyzz1/pro261004/cc0textures-512 \
    --num_scenes 100 --frames 1 --device CPU \
    > /home/xyyzz1/pro261004/s2_gen_100.log 2>&1

# ② 转成 images/ + annotations.json（8 角点）+ bop_raw/
python /home/xyyzz1/pro261004/s2_2_bop_to_corners.py /home/xyyzz1/pro261004/my_dataset \
    --out_dir /home/xyyzz1/pro261004/my_dataset_corners
```

### 1000 张图（`my_dataset1000/`）

1000 张 = 约 2.5~3 小时（CPU ~10s/张）。为避免单进程内存随场景数累积导致 OOM
（实测 100 场景 ≈ 3GB unfreed），**拆成 10 批 × 100 张顺序执行**（每批独立进程、内存重置；
BOP 帧自动 `append` 到同一 `train_pbr/000000/`）。脚本 `run_s2_1000.sh`：

```bash
# ① 建 BOP 工作目录
mkdir -p my_dataset1000_bop/models
cp my_dataset/models/obj_000001.ply  my_dataset1000_bop/models/
cp my_dataset/models/models_info.json my_dataset1000_bop/models/
cp my_dataset/camera.json             my_dataset1000_bop/

# ② 后台跑 10 批（run_s2_1000.sh 内部循环 10 次 --num_scenes 100 --frames 1）
bash /home/xyyzz1/pro261004/run_s2_1000.sh > s2_gen_1000.log 2>&1

# ③ 转成最终数据集 my_dataset1000/（images/ + annotations.json + bop_raw/）
python /home/xyyzz1/pro261004/s2_2_bop_to_corners.py /home/xyyzz1/pro261004/my_dataset1000_bop \
    --out_dir /home/xyyzz1/pro261004/my_dataset1000
```

## 10. 重要注意事项

1. **GPU 架构不匹配**：本机 RTX 5060 是 Blackwell（sm_120），而 Blender 3.6 的 Cycles CUDA
   内核最高只支持到 Ada（sm_89），CUDA 渲染会失败/回退。**冒烟测试用 `--device CPU`**。
   正式 GPU 加速需改用 bpy 4.x（需 Python 3.11 环境 + 适配 blenderproc 3.x→4.x API），
   或接受 CPU 渲染（慢）。
2. **单位**：BOP 模型必须是 mm。原始 `osmo_action4.ply` 是 cm，已用 `--scale 10` 处理。
3. **残留数据**：`write_bop` 默认 `append_to_existing_output=True`，若上次运行中断留下
   不完整的 `train_pbr/`（缺 `scene_gt_info.json`），下次运行会因尝试加载旧标注而报错。
   重跑前请 `rm -rf my_dataset/train_pbr`。
4. **s1_p2（KASAL 对称性 GUI）已跳过**：非必需；仅旋转对称物体需要 `_sym_type.json`。
5. **blenderproc** 为项目自带 fork（`/home/xyyzz1/pro261004/blenderproc/`），运行时需
   `PYTHONPATH=/home/xyyzz1/pro261004/blenderproc`（或 `pip install -e blenderproc`）。
6. **镜像**：全程清华 PyPI（~4MB/s），官方源太慢（45kB/s）。

---

*记录时间：2026-10-05*
