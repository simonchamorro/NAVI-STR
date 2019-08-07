import os
from tqdm import tqdm

images_path = './PyTorch-YOLOv3/data/sevn/images/'
empty_images_path = "./PyTorch-YOLOv3/data/sevn/images_noobj/"
data_path = './PyTorch-YOLOv3/data/sevn/'

test = []
train = []

with open(f'{data_path}train.txt', 'r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        if line.split('/')[-1] is not None:
            train.append(line.split('/')[-1].replace('\n', ''))
f.close()
with open(f'{data_path}valid.txt', 'r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        if line.split('/')[-1] is not None:
            test.append(line.split('/')[-1].replace('\n', ''))
f.close()

print(train)
i = 0
for img in tqdm(os.listdir(images_path), desc="Moving empty images to another folder"): 
    frame = img.split('/')[-1]
    if not frame in test and not frame in train:
        os.rename(images_path + img, empty_images_path + frame)
        i += 1

print(f'Moved {i} images that did not contain objects')
