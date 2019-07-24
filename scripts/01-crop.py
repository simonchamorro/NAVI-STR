"""Crops 16 views from equirrectangular panos"""

import os
import cv2
import argparse
import multiprocessing as mp
import numpy as np
import pandas as pd
from multiprocessing import Queue

from datetime import datetime
startTime = datetime.now()

def crop_proc(q):
    while True:
        try:
            fname = q.get(True, 10000000000)
            print("Processing: " + fname)
            frame_num = fname.split(".")[0].split("_")[-1]
            if do_img:
                img = cv2.imread(panos_path + fname)
                img = img[crop_margin:H - crop_margin]

            for idx in range(16):
                x = int(idx * W / 16)
                y = 0
                out_fname = crops_path + f'{frame_num}_{idx + 1}.png'
                if do_img and not os.path.isfile(out_fname):
                    crop_img(img, x, y, out_fname)
        except:
            return

def crop_img(img, x, y, out_fname):
    if (x + crop_W) % img.shape[1] != (x + crop_W):
        crop_img = np.zeros((crop_H, crop_W, 3))
        offset = img.shape[1] - (x % img.shape[1])
        crop_img[:, :offset] = img[y:y+crop_H, x:x + offset]
        crop_img[:, offset:] = img[y:y+crop_H, :(x + crop_W) % img.shape[1]]
    else:
        crop_img = img[:, x:x + crop_W]
    cv2.imwrite(out_fname, crop_img)

def remap_label():
    return 0


panos_path = "./data/panos/"
crops_path = "./data/crops/"
labels_path = "./data/labels/"
meta_df = pd.read_hdf(labels_path + "meta.hdf5", key="df", index=False)

W = int(3840)
H = int(1920)
crop_W = int(1280)
crop_H = int(1280)
crop_margin = int(H/6)
total_frames = len(os.listdir(panos_path))
q = Queue()

num_procs = 1
do_img = True
procs = []
new_labels = []

# import pdb; pdb.set_trace()
# labels = meta_df[meta_df.frame == 2271]
# for idx in range(len(labels)):
#     label = labels.iloc[idx]
#     x_min = label.x_min * 

for i in range(num_procs):
    p = mp.Process(target=crop_proc, args=(q,))
    procs.append(p)
    p.start()

fnames = [fname for fname in os.listdir(panos_path) if fname.split('.')[-1] == "png"]
while fnames:
    if q.empty():
        q.put(fnames.pop())

for p in procs:
    p.join()

print(datetime.now() - startTime)
