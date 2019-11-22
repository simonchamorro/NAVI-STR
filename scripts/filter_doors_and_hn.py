import os
from tqdm import tqdm
from shutil import copyfile
import pandas as pd
from nfov import NFOV
import imageio as im
import numpy as np
import cv2

nfov = NFOV()
images_path = '/home/martin/code/NAVI-STR/data/panos/'
dest = "./PyTorch-YOLOv3/data/sevn/images_doors_hn/"

test = []
train = []
labels_file = "/home/martin/code/SEVN/SEVN_gym/data/label.hdf5"
labels_df = pd.read_hdf(labels_file, key="df", index=False)

labels = labels_df[labels_df['obj_type'] != 'street_sign']

paths = []
for frame in labels.frame:
    paths.append("pano_" + str(frame).zfill(6) + ".png")

i = 0
for img in tqdm(os.listdir(images_path), desc="Moving empty images to another folder"): 
    frame = img.split('_')[1].split('.')[0]

    for idx, label in labels[labels.frame == int(frame)].iterrows():
        ratio = (3840 / 226)
        dim = (500, 800)
        center = int(round((label.x_max * ratio - label.x_min * ratio) / 2 + label.x_min * ratio)) / 3840
        center_point = np.array([center, .5])  # camera center point (valid range [0,1])
        # rimg = nfov.toNFOV(im.imread(images_path + img), center_point)
        # im.imwrite(dest + img, rimg)


    #
    # if frame in paths:
    #     copyfile(images_path + frame, doors_and_hn_images_path + frame)
    #     i += 1

print(f'Moved {i} images that contain doors or house numbers')
