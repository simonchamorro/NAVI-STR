import os
from tqdm import tqdm
from PIL import Image
import random
import subprocess
from collections import defaultdict


seed = 0
random.seed(seed)
pkg_path = "./ocr/"
train_lines = []
valid_lines = []
test_lines = []
train_hn = []
valid_hn = []
test_hn = []
train_ss = []
valid_ss = []
test_ss = []

house_numbers = defaultdict(list)
street_signs = defaultdict(list)

with open(f'{pkg_path}data/house_number_gt.txt', 'r') as f:
	for line in f:
		gt = line.split('\t')[-1].replace('\n', '')
		house_numbers[gt].append(line)

# with open(f'{pkg_path}data/street_sign_gt.txt', 'r') as f:
# 	for line in f:
# 		gt = line.split('\t')[-1].replace('\n', '')
# 		street_signs[gt].append(line)

gt_house_numbers = [x for x in house_numbers.keys()]
# gt_street_signs = [x for x in street_signs.keys()]

num_train_hn = int(0.8 * len(gt_house_numbers))
num_valid_hn = int(0.1 * len(gt_house_numbers))
num_test_hn = int(0.1 * len(gt_house_numbers))

# num_train_ss = int(0.8 * len(gt_street_signs))
# num_valid_ss = int(0.1 * len(gt_street_signs))
# num_test_ss = int(0.1 * len(gt_street_signs))

train_hn.extend(gt_house_numbers[:num_train_hn])
valid_hn.extend(gt_house_numbers[num_train_hn:num_train_hn + num_valid_hn])
test_hn.extend(gt_house_numbers[num_train_hn + num_valid_hn:])

# train_ss.extend(gt_street_signs[:num_train_ss])
# valid_ss.extend(gt_street_signs[num_train_ss:num_train_ss + num_valid_ss])
# test_ss.extend(gt_street_signs[num_train_ss + num_valid_ss:])

train_lines.extend([house_numbers[hn] for hn in train_hn])
# train_lines.extend([street_signs[ss] for ss in train_ss])
valid_lines.extend([house_numbers[hn] for hn in valid_hn])
# valid_lines.extend([street_signs[ss] for ss in valid_ss])
test_lines.extend([house_numbers[hn] for hn in test_hn])
# test_lines.extend([street_signs[ss] for ss in test_ss])

train_lines = [item for sublist in train_lines for item in sublist]
valid_lines = [item for sublist in valid_lines for item in sublist]
test_lines = [item for sublist in test_lines for item in sublist]

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
