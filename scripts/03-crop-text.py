"""Crops text bounding box from images"""

import os
import cv2
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


def crop_text(img_path):
    frame_crop = img_path.split('.')[0]
    frame = frame_crop.split('_')[0]
    frame_crop_num = int(frame_crop.split('_')[0]) + float(frame_crop.split('_')[1]) / 100
    if not frame_crop_num in labels_df['frame']:
        return

    labels = labels_df[labels_df.frame == frame_crop_num]
    for idx in range(len(labels)):
        label = labels.iloc[idx]
        if label.obj_type == 'door': continue
        elif label.obj_type == 'house_number':
            text = label.house_number
            out_fname = f'data/images/{text}_{label.street_name}_{frame}.png'
        elif label.obj_type == 'street_sign':
            text = label.street_name
            out_fname = f'data/images/{text}_{frame}.png'
        if out_fname in paths: continue
    
        img = cv2.imread(images_path + img_path)
        crop = img[label.y_min:label.y_max, label.x_min:label.x_max]
        if 0 in crop.shape: continue 
        
        # cv2.imwrite(dest_path + out_fname, crop)
        paths.append(out_fname)
        gt.append((text, label.obj_type))

images_path = "./PyTorch-YOLOv3/data/sevn/images/"
dest_path = "./deep-text-recognition-benchmark/"
labels_file = "./data/labels/crop_labels.hdf5"

labels_df = pd.read_hdf(labels_file, key="df", index=False)

total_imgs = len(os.listdir(images_path))
images = os.listdir(images_path)
paths = []
gt = []

for img_path in tqdm(images, desc="Croping images"):
    crop_text(img_path)

with open(f'{dest_path}data/gt.txt', 'w') as f:
    for idx in range(len(paths)):
        f.write(f"{paths[idx]}\t{gt[idx][0]}\n")
f.close()

with open(f'{dest_path}data/house_number_gt.txt', 'w') as f:
    for idx in range(len(paths)):
        if gt[idx][1] == 'house_number':
            f.write(f"{paths[idx]}\t{gt[idx][0]}\n")
f.close()

with open(f'{dest_path}data/street_sign_gt.txt', 'w') as f:
    for idx in range(len(paths)):
        if gt[idx][1] == 'street_sign':
            f.write(f"{paths[idx]}\t{gt[idx][0]}\n")
f.close()



