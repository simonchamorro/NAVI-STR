import os
from tqdm import tqdm
from PIL import Image
import random
import subprocess


seed = 0
random.seed(seed)
pkg_path = "./deep-text-recognition-benchmark/"
lines = []
train_lines = []
valid_lines = []
test_lines = []

for label_class in ['house_number', 'street_sign']:
	with open(f'{pkg_path}data/{label_class}_gt.txt', 'r') as f:
		for line in f:
			lines.append(line)

	num_train = int(0.8 * len(lines))
	num_valid = int(0.1 * len(lines))
	num_test = int(0.1 * len(lines))

	train_lines.extend(lines[:num_train])
	valid_lines.extend(lines[num_train:num_train + num_valid])
	test_lines.extend(lines[num_train + num_valid:])

random.shuffle(train_lines)
random.shuffle(valid_lines)
random.shuffle(test_lines)

output_train_path = f'{pkg_path}data/gt_train.txt'
output_valid_path = f'{pkg_path}data/gt_valid.txt'
output_test_path = f'{pkg_path}data/gt_test.txt'
with open(output_train_path, 'w') as f:
	for line in train_lines:
		f.write(line)
with open(output_valid_path, 'w') as f:
	for line in valid_lines:
		f.write(line)
with open(output_test_path, 'w') as f:
	for line in test_lines:
		f.write(line)

subprocess.call(f"python3 {pkg_path}create_lmdb_dataset.py --inputPath {pkg_path} data/ --gtFile {output_train_path} --outputPath {pkg_path}data/train/", shell=True)
subprocess.call(f"python3 {pkg_path}create_lmdb_dataset.py --inputPath {pkg_path} data/ --gtFile {output_valid_path} --outputPath {pkg_path}data/valid/", shell=True)
subprocess.call(f"python3 {pkg_path}create_lmdb_dataset.py --inputPath {pkg_path} data/ --gtFile {output_test_path} --outputPath {pkg_path}data/test/", shell=True)
