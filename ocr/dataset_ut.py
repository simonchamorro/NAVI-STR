import os

from collections import Counter, OrderedDict
from easydict import EasyDict as edict
import numpy as np

from dataset import hierarchical_dataset, AlignCollate

import matplotlib.pyplot as plt


def bar_plot_from_dict(d, fname, xlabel, ylabel):
    plt.bar(range(len(d)), list(d.values()), align='center')
    plt.xticks(range(len(d)), list(d.keys()))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(fname)
    plt.close()


def histogram_from_list(l, fname, xlabel, ylabel, threshold):
    plt.hist(l, bins=100)
    plt.axvline(x=threshold, color='r')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(fname)
    plt.close()


def scatter_from_lists(imgs_w, imgs_h, fname, xlabel, ylabel,
                       x_threshold, y_threshold):
    plt.scatter(imgs_w, imgs_h)
    plt.axvline(x=x_threshold, color='r')
    plt.axhline(y=y_threshold, color='r')
    # Count samples in comparison to the thresholds
    # Bottom left
    bl_txt = str(len([img for img in list(zip(imgs_w, imgs_h)) if
                      (img[0] < x_threshold and img[1] < y_threshold)]))
    plt.annotate(bl_txt,
                 ((x_threshold-min(imgs_w))/2., (y_threshold-min(imgs_h))/2.))
    # Top left
    tl_txt = str(len([img for img in list(zip(imgs_w, imgs_h)) if
                      (img[0] < x_threshold and img[1] > y_threshold)]))
    plt.annotate(tl_txt,
                 ((x_threshold-min(imgs_w))/2., (max(imgs_h)-y_threshold)/2.))
    # Bottom right
    br_txt = str(len([img for img in list(zip(imgs_w, imgs_h)) if
                      (img[0] > x_threshold and img[1] < y_threshold)]))
    plt.annotate(br_txt,
                 ((max(imgs_w)-x_threshold)/2., (y_threshold-min(imgs_h))/2.))
    # Top right
    tr_txt = str(len([img for img in list(zip(imgs_w, imgs_h)) if
                      (img[0] > x_threshold and img[1] > y_threshold)]))
    plt.annotate(tr_txt,
                 ((max(imgs_w)-x_threshold)/2., (max(imgs_h)-y_threshold)/2.))
    # Labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(fname)
    plt.close()


def save_imgs(img_rgb, img_grey, input_nopad, input_pad, label, fname):
    f, axarr = plt.subplots(2, 2)
    # RGB image
    axarr[0, 0].imshow(np.asarray(img_rgb), vmin=0, vmax=255)
    axarr[0, 0].set_title('RGB image')
    # Grayscale image
    axarr[0, 1].imshow(np.asarray(img_grey), cmap='gray', vmin=0, vmax=255)
    axarr[0, 1].set_title('Grey image')
    # Transformed grayscale image without padding
    axarr[1, 0].imshow(input_nopad, cmap='gray', vmin=-1, vmax=1)
    axarr[1, 0].set_title('Input without padding')
    # Transformed grayscale image with padding
    axarr[1, 1].imshow(input_pad, cmap='gray', vmin=-1, vmax=1)
    axarr[1, 1].set_title('Input with padding')
    f.suptitle(label)
    plt.savefig(fname)
    plt.close()


if __name__ == '__main__':
    # Set dataset/dataloader options
    opt = edict({})
    # Label preprocessing
    opt.predict_number = True
    opt.label_filter_type = 'leq'  # or 'eq'
    opt.batch_max_length = 4
    opt.sensitive = False
    # Image preprocessing
    opt.imgH = 32
    opt.imgW = 100

    data_sets = ['train', 'valid', 'test']
    labels_sets = []
    for data_set in data_sets:
        print(f'Data set: {data_set}')
        image_output = f'image_test/{data_set}'
        print(f'image_output: {image_output}')
        if not os.path.exists(image_output):
            os.makedirs(image_output)
        if not os.path.exists(f'{image_output}/imgs'):
            os.makedirs(f'{image_output}/imgs')

        # Construct datasets
        # RGB
        opt.rgb = True
        rgb_dataset = hierarchical_dataset(root=f'data/{data_set}',
                                           opt=opt,
                                           select_data=['data'])
        # Greyscale
        opt.rgb = False
        dataset = hierarchical_dataset(root=f'data/{data_set}',
                                       opt=opt,
                                       select_data=['data'])

        assert len(rgb_dataset) == len(dataset)
        print(f'num total {set} samples: {len(dataset)}')

        imgs_rgb = []
        imgs = []
        inputs = []
        inputs_pad = []
        labels = []
        numbers = []
        for index in range(len(rgb_dataset)):
            # RGB
            opt.rgb = True
            (img_rgb, rgb_label) = rgb_dataset[index]

            # Greyscale
            opt.rgb = False
            (img, label) = dataset[index]
            assert rgb_label == label

            # Input without padding
            opt.keep_ratio_with_pad = False
            AlignCollate_nopad = AlignCollate()
            image_tensor, label_align = AlignCollate_nopad([(img, label)])
            assert label_align[0] == label
            img_tensor = image_tensor[0, 0, :, :]
            img_tensor = img_tensor.cpu().float().numpy()

            # Input with padding
            opt.keep_ratio_with_pad = True
            AlignCollate_pad = AlignCollate()
            image_tensor, label_align = AlignCollate_pad([(img, label)])
            assert label_align[0] == label
            img_tensor_pad = image_tensor[0, 0, :, :]
            img_tensor_pad = img_tensor_pad.cpu().float().numpy()

            imgs_rgb.append(img_rgb)
            imgs.append(img)
            inputs.append(img_tensor)
            inputs_pad.append(img_tensor_pad)
            labels.append(label)
            numbers.append(label.split('_')[0])

        labels_sets.append(labels)

        # Stats about sequence lenght
        unique_labels = set(labels)
        unique_numbers = [label.split('_')[0] for label in unique_labels]
        print(f'num of unique labels: {len(unique_labels)}')

        seq_lenght = [len(nb) for nb in numbers]
        unique_seq_lenght = [len(nb) for nb in unique_numbers]

        d_seq_lenght = Counter(seq_lenght)
        d_unique_seq_lenght = Counter(unique_seq_lenght)

        bar_plot_from_dict(OrderedDict(sorted(d_seq_lenght.items())),
                           f'{image_output}/seq_lenght.png',
                           'sequence lenght',
                           'frequency')
        bar_plot_from_dict(OrderedDict(sorted(d_unique_seq_lenght.items())),
                           f'{image_output}/unique_seq_lenght.png',
                           'unique sequence lenght',
                           'frequency')

        # Stats about digits when taken individually
        ind_digit = [item for sublist in numbers for item in sublist]
        ind_digit_unique = [item for sublist in unique_numbers for item in sublist]

        d_ind_digit = Counter(ind_digit)
        d_ind_digit_unique = Counter(ind_digit_unique)

        bar_plot_from_dict(OrderedDict(sorted(d_ind_digit.items())),
                           f'{image_output}/ind_digit.png',
                           'digit',
                           'frequency')
        bar_plot_from_dict(OrderedDict(sorted(d_ind_digit_unique.items())),
                           f'{image_output}/ind_digit_unique.png',
                           'digit (for unique numbers)',
                           'frequency')

        # Stats about number of images per unique address
        groub_by_address = Counter(labels)
        nb_views = groub_by_address.values()
        d_nb_views = Counter(nb_views)

        bar_plot_from_dict(OrderedDict(sorted(d_nb_views.items())),
                           f'{image_output}/ind_digit_unique.png',
                           'number of views',
                           'frequency')

        # Take a look to original image shapes
        imgs_shape = [img.size for img in imgs]
        print(f'number of unique shape: {len(set(imgs_shape))}')
        imgs_w = [shape[0] for shape in imgs_shape]
        imgs_h = [shape[1] for shape in imgs_shape]

        histogram_from_list(imgs_w,
                            f'{image_output}/imgs_w.png',
                            'width',
                            'frequency',
                            opt.imgW)

        histogram_from_list(imgs_h,
                            f'{image_output}/imgs_h.png',
                            'height',
                            'frequency',
                            opt.imgH)

        scatter_from_lists(imgs_w, imgs_h,
                           f'{image_output}/scatter_imgs_w_h.png',
                           'width',
                           'height',
                           opt.imgW,
                           opt.imgH)

        # Plot images and label
        for index in range(len(imgs)):
            img_rgb = imgs_rgb[index]
            img_grey = imgs[index]
            input_nopad = inputs[index]
            input_pad = inputs_pad[index]
            label = labels[index]
            street_segment = label.split('_')[1]
            number = label.split('_')[0]
            dir_fname = f'{image_output}/imgs/{street_segment}/{number}/'
            if not os.path.exists(dir_fname):
                os.makedirs(dir_fname)
            fname = dir_fname + f'{label}_index{index}.png'
            print(fname)
            save_imgs(img_rgb, img_grey, input_nopad, input_pad, label, fname)

    # Check intersection between data sets
    assert not bool(set(labels_sets[0]) & set(labels_sets[1]))
    assert not bool(set(labels_sets[0]) & set(labels_sets[2]))
    assert not bool(set(labels_sets[1]) & set(labels_sets[2]))
