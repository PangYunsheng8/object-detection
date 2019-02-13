# -*- coding: utf-8 -*-
# @Author: Yongqiang Qin
# @Date:   2018-05-02 19:26:55
# @Last Modified by:   Yongqiang Qin
# @Last Modified time: 2018-05-15 18:53:14

import os
import sys
import argparse

ffmpeg = 'ffmpeg'

def cut_videos(video_folder, img_folder, fps=2):
    for root, dirs, files in os.walk(video_folder):
        files = [_ for _ in files if _.endswith(('mp4', 'MP4', 'MOV')) and not _.startswith(".")]

        for f in files:
            img_prefix = os.path.join(img_folder, f[:-4])
            if not os.path.exists(img_prefix):
                os.makedirs(img_prefix)
            img_prefix = os.path.join(img_prefix, '%04d.jpg')
            video_path = os.path.join(root, f)

            cmd = ' '.join([ffmpeg, '-i', '"'+video_path+'"', '-vf', 'fps={}'.format(fps), '"'+img_prefix+'"', '-hide_banner'])
            # cmd = ' '.join([ffmpeg, '-i', '"'+video_path+'"', '-vf', '"select=eq(pict_type\,I)"', '-vsync', 'vfr', '"'+img_prefix+'"', '-hide_banner'])
            os.system(cmd)
            # print(cmd)
            # exit(0)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_folder", help="Video folder contains videos endswith mp4, MP4 or MOV.")
    parser.add_argument("frame_folder", help="The output folder of vidoes frames cut from videos.")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    if not os.path.exists(args.video_folder):
        os.makedirs(args.video_folder)

    cut_videos(args.video_folder, args.frame_folder, fps=4)
