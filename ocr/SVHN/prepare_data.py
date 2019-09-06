import os

import h5py
from PIL import Image
from tqdm import tqdm
import numpy as np

from boxes import extract_labels_boxes, extract_outer_box
from misc import save_obj
from visualization import visualize_sample


def get_box_data(index, hdf5_data):
    '''
    Source:
    https://stackoverflow.com/questions/41176258/h5py-access-data-in-datasets-in-svhn
    Get left, top, width, height' of each picture.
    '''
    meta_data = dict()
    meta_data['height'] = []
    meta_data['label'] = []
    meta_data['left'] = []
    meta_data['top'] = []
    meta_data['width'] = []

    def print_attrs(name, obj):
        vals = []
        if obj.shape[0] == 1:
            vals.append(obj[0][0])
        else:
            for k in range(obj.shape[0]):
                vals.append(int(hdf5_data[obj[k][0]][0][0]))
        meta_data[name] = vals

    box = hdf5_data['/digitStruct/bbox'][index]
    hdf5_data[box[0]].visititems(print_attrs)
    return meta_data


def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[(name[index][0])]])


if __name__ == '__main__':
    check_valid = True
    valid_split = 0.2

    splits = ['train', 'extra', 'test']
    for split in splits:

        # Create output dir
        output_dir = f'{split}_crop'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(f'{split}_visualize_sample'):
            os.makedirs(f'{split}_visualize_sample')

        # Load metadata
        fname = f'{split}/digitStruct.mat'
        mat_data = h5py.File(fname)
        dataset_size = mat_data['/digitStruct/name'].size
        print(f'{split} dataset size: {dataset_size}')

        if split in ['train', 'extra']:
            train_dataset_size = int((1 - valid_split) * dataset_size)
            valid_dataset_size = dataset_size - train_dataset_size
            valid_output_dir = f'validfrom{split}_crop'
            if not os.path.exists(valid_output_dir):
                os.makedirs(valid_output_dir)
            indexes = [range(train_dataset_size),
                       range(train_dataset_size, dataset_size)]
            outputs = [output_dir, valid_output_dir]
        else:
            indexes = [range(dataset_size)]
            outputs = [output_dir]

        for i, index in enumerate(indexes):
            # Create process dataset
            with open(f'{outputs[i]}/labels.txt', 'w') as x_file:
                for idx in tqdm(index):
                    # Image path
                    img_id = get_name(idx, mat_data)
                    img_name = f'{split}/{img_id}'
                    # Load image
                    img = Image.open(img_name)

                    # Labels
                    labels = get_box_data(idx, mat_data)
                    # Convert label 10 to label 0 for digit 0
                    if 10 in labels['label']:
                        labels['label'] = [0 if x == 10 else x for x in
                                           labels['label']]
                    labels, boxes = extract_labels_boxes(labels)
                    outer_bbox = extract_outer_box(img, boxes, padding=0.3)

                    number = ''.join([str(nb) for nb in labels])
                    x_file.write(f'{img_id}\t{number}\n')

                    if index in range(10):
                        file = f'{split}_visualize_sample/visualize_{img_id}'
                        visualize_sample(np.array(img), boxes, labels,
                                         file,
                                         outer_bbox)

                    # Cropped image
                    x1, x2, y1, y2 = outer_bbox
                    img_crop = img.crop((x1, y1, x2, y2))
                    img_crop.save(f'{outputs[i]}/{img_id}')
                x_file.close()
