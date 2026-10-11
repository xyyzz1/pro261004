#!/bin/bash
# 生成 1000 张实验数据：拆成 10 个批次 × 每批 100 张，顺序执行。
# 每批独立进程（内存重置，避免单进程长期累积导致 OOM），BOP 帧自动追加到同一 train_pbr/。
set -e
source /home/xyyzz1/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /home/xyyzz1/anaconda3/etc/profile.d/conda.sh
conda activate pro261004
cd /home/xyyzz1/pro261004/my_dataset1000_bop
export PYTHONPATH=/home/xyyzz1/pro261004/blenderproc

for b in $(seq 1 10); do
    echo "===== batch $b/10 start : $(date '+%H:%M:%S') ====="
    python /home/xyyzz1/pro261004/s2_p1_gen_pbr_data.py \
        0 /home/xyyzz1/pro261004/cc0textures-512 \
        --num_scenes 100 --frames 1 --device CPU
    echo "===== batch $b/10 done  : $(date '+%H:%M:%S') ====="
done
echo "===== ALL 10 BATCHES DONE ====="
