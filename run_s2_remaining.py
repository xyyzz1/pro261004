#!/usr/bin/env python
# 以较小单批规模续跑剩余数据：每批最多 BATCH 张，循环到 TARGET 张。
# 每批结束重新统计已完成帧数（取 scene_gt.json 与 scene_gt_info.json 的最小值），
# 避免中断/残留导致统计偏大；最后一批自动按剩余量截断。
import json
import os
import sys
import subprocess

DATASET = '/home/xyyzz1/pro261004/my_dataset1000_bop'
SCENE_DIR = os.path.join(DATASET, 'train_pbr', '000000')
BATCH = 40
TARGET = 1000
SCRIPT = '/home/xyyzz1/pro261004/s2_p1_gen_pbr_data.py'
CCTEX = '/home/xyyzz1/pro261004/cc0textures-512'
BLENDERPROC = '/home/xyyzz1/pro261004/blenderproc'


def count_done():
    def nframes(name):
        p = os.path.join(SCENE_DIR, name)
        if not os.path.exists(p):
            return 0
        try:
            return len(json.load(open(p)))
        except Exception:
            return 0
    # 只有 scene_gt 与 scene_gt_info 都写全的帧才算“完成”，避免残留半帧。
    return min(nframes('scene_gt.json'), nframes('scene_gt_info.json'))


def main():
    while True:
        n = count_done()
        if n >= TARGET:
            print('REACHED %d >= %d, done' % (n, TARGET), flush=True)
            break
        rem = min(BATCH, TARGET - n)
        print('[%d/%d] running %d scenes : %s' % (n, TARGET, rem, __import__('datetime').datetime.now().strftime('%H:%M:%S')), flush=True)
        env = dict(os.environ)
        env['PYTHONPATH'] = BLENDERPROC
        cmd = [sys.executable, SCRIPT, '0', CCTEX,
               '--num_scenes', str(rem), '--frames', '1', '--device', 'CPU']
        subprocess.run(cmd, cwd=DATASET, check=True, env=env)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
