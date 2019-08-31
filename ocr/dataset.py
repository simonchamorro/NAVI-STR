import os
import sys
import six
import math
import lmdb
import torch

from natsort import natsorted
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms


class Batch_Balanced_Dataset(object):

    def __init__(self, opt):
        '''
        Construct the batch balanced dataset.
        Modulate the data ratio in the batch.
        For example, when select_data is "MJ-ST" and batch_ratio is "0.5-0.5",
        the 50% of the batch is filled with MJ and the other 50% of the batch
        is filled with ST.

        Args:
            opt (argparse.Namespace): Arguments to use for the experiment.
        '''
        print('-' * 80)
        print(f'dataset_root: {opt.train_data} ')
        print(f'opt.select_data: {opt.select_data}')
        print(f'opt.batch_ratio: {opt.batch_ratio}')
        assert len(opt.select_data) == len(opt.batch_ratio)
        _AlignCollate = AlignCollate(imgH=opt.imgH,
                                     imgW=opt.imgW,
                                     keep_ratio_with_pad=opt.PAD)
        self.data_loader_list = []
        self.dataloader_iter_list = []
        batch_size_list = []
        Total_batch_size = 0
        # Construct the dataset and dataloader for each sub-dataset.
        for selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio):
            _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)
            print('-' * 80)
            _dataset = hierarchical_dataset(root=opt.train_data, opt=opt,
                                            select_data=[selected_d])
            total_number_dataset = len(_dataset)

            '''
            The total number of data can be modified with
            opt.total_data_usage_ratio.
            ex) opt.total_data_usage_ratio = 1 indicates 100% usage, and 0.2
            indicates 20% usage. See 4.2 section in our paper.
            '''
            number_dataset = int(total_number_dataset *
                                 float(opt.total_data_usage_ratio))
            dataset_split = [number_dataset,
                             total_number_dataset - number_dataset]
            indices = range(total_number_dataset)
            _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                           for offset, length in
                           zip(_accumulate(dataset_split), dataset_split)]
            print(f'num total samples of {selected_d}:' +
                  f' {total_number_dataset} x {opt.total_data_usage_ratio}' +
                  f' (total_data_usage_ratio) = {len(_dataset)}')
            print(f'num samples of {selected_d} per batch:' +
                  f' {opt.batch_size} x {float(batch_ratio_d)}' +
                  f' (batch_ratio) = {_batch_size}')

            batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            _data_loader = torch.utils.data.DataLoader(
                _dataset,
                batch_size=_batch_size,
                shuffle=True,
                num_workers=int(opt.workers),
                collate_fn=_AlignCollate,  # Resize and nomalize
                pin_memory=True)
            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))

        print('-' * 80)
        print('Total_batch_size: ',
              '+'.join(batch_size_list), '=', str(Total_batch_size))
        opt.batch_size = Total_batch_size
        print('-' * 80)

    def get_batch(self):
        '''
        Get batch.

        Returns:
            balanced_batch_images (tensor): Batch of images of size
            [batch_size, c, h, w].
            balanced_batch_texts (list): List of texts of len batch_size.
        '''
        balanced_batch_images = []
        balanced_batch_texts = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, text = data_loader_iter.next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, text = self.dataloader_iter_list[i].next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except ValueError:
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)
        return balanced_batch_images, balanced_batch_texts


def hierarchical_dataset(root, opt, select_data='/'):
    '''
    Constuct hierarchical dataset from one or several LMDB database
    (https://en.wikipedia.org/wiki/Lightning_Memory-Mapped_Database).

    Args:
        root (str): path to root directory.
        opt (argparse.Namespace): Arguments to use for the experiment.
        select_data (list): list of data directories.
        select_data='/' contains all sub-directory of root directory.

    Returns:
        concatenated_dataset (object): Dataset or concatenate dataset if
        several sub-directory in select_data.
    '''
    dataset_list = []
    print(f'dataset_root:    {root}\t dataset: {select_data[0]}')
    for dirpath, dirnames, filenames in os.walk(root):
        if not dirnames:
            select_flag = False
            for selected_d in select_data:
                if selected_d in dirpath:
                    select_flag = True
                    break

            if select_flag:
                print(f'dir_path: {dirpath}')
                dataset = LmdbDataset(dirpath, opt)
                print(f'sub-directory:' +
                      f'\t/{os.path.relpath(dirpath, root)}\t' +
                      f' num samples: {len(dataset)}')
                dataset_list.append(dataset)

    concatenated_dataset = ConcatDataset(dataset_list)
    return concatenated_dataset


class LmdbDataset(Dataset):

    def __init__(self, root, opt):
        '''
        Build dataset from an LMDB database
        (https://en.wikipedia.org/wiki/Lightning_Memory-Mapped_Database).

        Args:
            root (str): Path to root directory.
            opt (argparse.Namespace): Arguments to use for the experiment.
        '''

        self.root = root
        self.opt = opt

        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False,
                             readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

            # Filtering
            self.filtered_index_list = []
            for index in range(self.nSamples):
                index += 1  # lmdb starts with 1
                label_key = 'label-%09d'.encode() % index
                label = txn.get(label_key).decode('utf-8')

                # opt.batch_max_length is the maximum label length, can be
                # set in train.py. The examples are filtered based on that.
                if len(label) > self.opt.batch_max_length:
                    print(f'The length of the label is longer than' +
                          f' max_length: length {len(label)}, {label}' +
                          f' in dataset {self.root}')
                    continue

                self.filtered_index_list.append(index)

            self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        '''
        Args:
            index (int): Index of the sample to get

        Returns:
            (img, label) (2-tuple): Contain the image (PIL.Image.Image) and the
            label (str).
        '''
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                if self.opt.rgb:
                    # For color image
                    img = Image.open(buf).convert('RGB')
                else:
                    # Load and eventually convert to greyscale
                    img = Image.open(buf).convert('L')

            except IOError:
                print(f'Corrupted image for {index}')
                # Make dummy image and dummy label for corrupted image
                if self.opt.rgb:
                    img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                else:
                    img = Image.new('L', (self.opt.imgW, self.opt.imgH))
                label = '[dummy_label]'

            if not self.opt.sensitive:
                # Lowecase the string
                label = label.lower()

            # We only train and evaluate on alphanumerics (or pre-defined
            # character set in train.py)
            # out_of_char = f'[^{self.opt.character}]'
            # import re
            # label = re.sub(out_of_char, '', label)

        return (img, label)


class RawDataset(Dataset):

    def __init__(self, root, opt):
        self.opt = opt
        self.image_path_list = []
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                _, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    self.image_path_list.append(os.path.join(dirpath, name))

        self.image_path_list = natsorted(self.image_path_list)
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        try:
            if self.opt.rgb:
                # For color image
                img = Image.open(self.image_path_list[index]).convert('RGB')
            else:
                img = Image.open(self.image_path_list[index]).convert('L')

        except IOError:
            print(f'Corrupted image for {index}')
            # Make dummy image and dummy label for corrupted image.
            if self.opt.rgb:
                img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
            else:
                img = Image.new('L', (self.opt.imgW, self.opt.imgH))

        return (img, self.image_path_list[index])


class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        '''
        Rezise and normalize by (img-0.5)/0.5 an image.

        Args:
            size (2-tuple): The requested size in pixels, as a 2-tuple:
            (width, height).
            interpolation (object): Resampling filter. Default Image.BICUBIC
            for cubic spline interpolation.

        Returns:
            img (tensor): The transformed image.

        '''
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        # Resize image
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        # Normalize image
        img.sub_(0.5).div_(0.5)
        return img


class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        '''
        Normalize by (img-0.5)/0.5 and pad an image.

        Args:
            max_size (3-tuple): The requested size in pixels, as a 3-tuple:
            (chanel, widht, height).
            PAD_tytpe (str): Padding type. Default: 'right'. Only option
            implemented.

        Returns:
            Pad_img (tensor): the transformed image.
        '''
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        # Normalize image
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        # Right pad with zeros
        Pad_img[:, :, :w] = img
        if self.max_size[2] != w:
            # Add right border Pad by expanding the image
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(
                c, h, self.max_size[2] - w)

        return Pad_img


class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False):
        '''
        Resize the image to imgW*imgH and normalise it by (img-0.5)/0.5.

        Args:
            imgH (int): The requested image height.
            imgW (int): The requested image width.
            keep_ratio_with_pad (bool): If True, keep the ratio of the original
            image and apply zero padding. Does not work with rgb images.

        Returns:
            image_tensors (tensor): The transformed images of size
            [batch_size, c, h, w].
            labels (tuple): The labels of len batch_size.
        '''
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            transform = NormalizePAD((1, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                # Resize the image
                resized_image = image.resize((resized_w, self.imgH),
                                             Image.BICUBIC)
                # Normalize + padding
                resized_images.append(transform(resized_image))
                # resized_image.save('./image_test/%d_test.jpg' % w)

            image_tensors = torch.cat(
                [t.unsqueeze(0) for t in resized_images], 0)

        else:
            # Resize and normalize
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat(
                [t.unsqueeze(0) for t in image_tensors], 0)
        return image_tensors, labels


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
