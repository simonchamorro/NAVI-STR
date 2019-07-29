"""Crops 16 views from equirrectangular panos and remaps their labels"""

import os
import cv2
import argparse
import multiprocessing as mp
import numpy as np
import pandas as pd
from multiprocessing import Queue
from tqdm import tqdm
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
            if y_min < 0 or y_max < 0: continue
            if y_max + 1 > crop_H: continue
            labels.append((frame, obj_type, house_number, street_name, x_min, x_max, y_min, y_max))
    label_df = pd.DataFrame(labels, columns = ["frame", "obj_type", "house_number", "street_name", "x_min", "x_max", "y_min", "y_max"])
    print("num labels failed to parse: " + str(len(failed_to_parse)))
    return label_df

def process(fname):
    # print("Processing: " + fname)
    frame_num = fname.split(".")[0].split("_")[-1]
    if do_img:
        img = cv2.imread(panos_path + fname)
        img = img[crop_margin:H - crop_margin]

    for i in range(16):
        x = int(i * W / 16)
        y = 0
        out_fname = crops_path + f'{frame_num}_{i + 1}.png'
        if do_img and not os.path.isfile(out_fname):
            crop_img(img, x, y, out_fname)
        if do_labels:
            process_frame_labels(frame_num, i, x)

def process_frame_labels(frame_num, i, x):
    labels = label_df[label_df.frame == int(frame_num)]
    temp = []
    for j in range(len(labels)):
        label_x_min = labels.iloc[j].x_min - x
        label_x_max = labels.iloc[j].x_max - x
        if label_x_min < 0: label_x_min = W - x + labels.iloc[j].x_min
        if label_x_max < 0: label_x_max = W - x + labels.iloc[j].x_max
        if label_x_max <= crop_W and label_x_min <= crop_W:
            frame = float(frame_num) + (i+1)/100
            temp.append((frame, labels.iloc[j].obj_type, labels.iloc[j].house_number, \
            labels.iloc[j].street_name, label_x_min, label_x_max, labels.iloc[j].y_min, labels.iloc[j].y_max))
     
    if len(temp) != 0:
        with open(f'{yolo_labels_path}{frame_num}_{i + 1}.txt', 'w') as f:
            for tup in temp:
                class_idx = class_to_idx(tup[1])
                x_center = ((tup[5] + tup[4]) / 2) / crop_W
                y_center = ((tup[7] + tup[6]) / 2) / crop_H
                width = (tup[5] - tup[4]) / crop_W
                height = (tup[7] - tup[6]) / crop_H
                f.write(f"{class_idx} {x_center} {y_center} {width} {height}\n")
        f.close()
    new_labels.extend(temp)

def class_to_idx(obj_type):
    if obj_type == "door":
        return 0
    elif obj_type == "house_number":
        return 1
    elif obj_type =="street_sign":
        return 2
    return -1

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
crops_path = "./PyTorch-YOLOv3/data/sevn/images/"
labels_path = "./data/labels/"
yolo_labels_path = "./PyTorch-YOLOv3/data/sevn/labels/"

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

do_img = False
do_labels = True
new_labels = []    

fnames = [fname for fname in os.listdir(panos_path) if fname.split('.')[-1] == "png"]
for fname in tqdm(fnames, desc="Processing panos"):
    process(fname)

new_label_df = pd.DataFrame(new_labels, columns = ["frame", "obj_type", "house_number", "street_name", "x_min", "x_max", "y_min", "y_max"])
label_index = new_label_df.groupby(new_label_df.frame).cumcount()
new_label_df.index = pd.MultiIndex.from_arrays([new_label_df.frame, label_index], names=["frame", "label"])
new_label_df.sort_index(inplace=True)
if do_labels:
    new_label_df.to_hdf(labels_path + "crop_labels.hdf5", key="df", index=False)
print(datetime.now() - startTime)
