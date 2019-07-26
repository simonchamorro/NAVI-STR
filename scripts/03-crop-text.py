"""Crops text bounding box from images"""

import os
import cv2
import argparse
import numpy as np
import pandas as pd


def crop_text(img_path):
    frame = img_path.split('.')[0]
    frame = int(frame.split('_')[0]) + float(frame.split('_')[1]) / 100
    if not frame in labels_df['frame']:
        return

    img = cv2.imread(images_path + img_path)
    labels = labels_df[labels_df.frame == frame]
    for idx in range(len(labels)):
        label = labels.iloc[idx]
        if label.obj_type == 'door': continue
        cv2.imshow('test', img[label.y_min:label.y_max, label.x_min:label.x_max])
        cv2.waitKey()


images_path = "./PyTorch-YOLOv3/data/sevn/images/"
crops_path = "./deep-text-recognition-benchmark/data/images/"
labels_file = "./data/labels/crop_labels.hdf5"

labels_df = pd.read_hdf(labels_file, key="df", index=False)

total_imgs = len(os.listdir(images_path))
images = os.listdir(images_path)

# import pdb; pdb.set_trace()
for img_path in images:
    crop_text(img_path)


