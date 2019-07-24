"""Crops 16 views from equirrectangular panos"""

import os
import cv2
import argparse
import multiprocessing as mp
import numpy as np
import pandas as pd
from multiprocessing import Queue
import xml.etree.ElementTree as et

from datetime import datetime
startTime = datetime.now()

def process_labels(paths):
    """ This function processes the labels into a nice format"""
    labels = []
    failed_to_parse = []
    for p in paths:
        xtree = et.parse(p)
        xroot = xtree.getroot()
        for idx, node in enumerate(xroot):
            if node.tag != "object":
                continue
            frame = int(p.split("_")[-1].split(".")[0])
            text_label = node.find("name").text
            house_number = None
            if text_label.split("-")[0] == "street_sign":
                try:
                    obj_type, street_name = text_label.split("-")
                except Exception as e:
                    # print("street_sign: " + str(e))
                    failed_to_parse.append(text_label)
                    continue
            elif text_label.split("-")[0] == "house_number":
                try:
                    obj_type, house_number, street_name = text_label.split("-")
                except Exception as e:
                    # print("house_number: " + str(e))
                    failed_to_parse.append(text_label)
                    continue
            elif text_label.split("-")[0] == "door":
                try:
                    obj_type, house_number, street_name = text_label.split("-")
                except Exception as e:
                    print("door: " + str(e))
                    failed_to_parse.append(text_label)
                    continue
            x_min = int(node.find("bndbox").find("xmin").text)
            x_max = int(node.find("bndbox").find("xmax").text)
            y_min = int(node.find("bndbox").find("ymin").text) - crop_margin
            y_max = int(node.find("bndbox").find("ymax").text) - crop_margin
            labels.append((frame, obj_type, house_number, street_name, x_min, x_max, y_min, y_max))
    label_df = pd.DataFrame(labels, columns = ["frame", "obj_type", "house_number", "street_name", "x_min", "x_max", "y_min", "y_max"])
    print("num labels failed to parse: " + str(len(failed_to_parse)))
    return label_df

def crop_proc(q):
    while True:
        try:
            fname = q.get(True, 1)
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


panos_path = "./data/panos/"
crops_path = "./data/crops/"
labels_path = "./data/labels/"

W = int(3840)
H = int(1920)
crop_W = int(1280)
crop_H = int(1280)
crop_margin = int(H/6)
total_frames = len(os.listdir(panos_path))
frames = [fname.split(".")[0].split("_")[-1] for fname in os.listdir(panos_path)]
paths = [labels_path + "raw/pano_" + str(frame).zfill(6) + ".xml" for frame in frames]
paths = [path for path in paths if os.path.isfile(path)]
label_df = process_labels(paths)
label_index = label_df.groupby(label_df.frame).cumcount()
label_df.index = pd.MultiIndex.from_arrays([label_df.frame, label_index], names=["frame", "label"])
label_df.sort_index(inplace=True)

q = Queue()

num_procs = 1
do_img = True
procs = []
new_labels = []

# import pdb; pdb.set_trace()
# labels = label_df[label_df.frame == 2271]
# for idx in range(len(labels)):
#     x_min = labels.iloc[idx].x_min - x
#     x_max = labels.iloc[idx].x_max - x
#     if x_min < 0: x_min = W - x + x_min
#     if x_max < 0: x_max = W - x + x_max
#     if x_max <= crop_W and x_max <= crop_W:
#         new_labels.append((frame, obj_type, house_number, street_name, x_min, x_max, y_min, y_max))
#     label_df = pd.DataFrame(labels, columns = ["frame", "obj_type", "house_number", "street_name", "x_min", "x_max", "y_min", "y_max"])
    

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
