import os

from easydict import EasyDict as edict

from dataset import hierarchical_dataset, AlignCollate

import matplotlib.pyplot as plt


def save_inputs(input, fname):
    # Transformed grayscale image with padding
    plt.imshow(input, cmap='gray', vmin=-1, vmax=1)
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
    for data_set in data_sets:
        print(f'Data set: {data_set}')
        image_output = f'image_check/{data_set}'
        print(f'image_output: {image_output}')
        if not os.path.exists(image_output):
            os.makedirs(image_output)

        # Construct datasets
        # Greyscale
        opt.rgb = False
        dataset = hierarchical_dataset(root=f'data/{data_set}',
                                       opt=opt,
                                       select_data=['data'])

        print(f'num total {set} samples: {len(dataset)}')

        inputs_pad = []
        labels = []
        numbers = []
        for index in range(len(dataset)):
            # Greyscale
            opt.rgb = False
            (img, label) = dataset[index]

            # Input with padding
            opt.keep_ratio_with_pad = True
            AlignCollate_pad = AlignCollate(opt.imgH, opt.imgW,
                                            opt.keep_ratio_with_pad)
            image_tensor, label_align = AlignCollate_pad([(img, label)])
            assert label_align[0] == label
            img_tensor_pad = image_tensor[0, 0, :, :]
            img_tensor_pad = img_tensor_pad.cpu().float().numpy()

            inputs_pad.append(img_tensor_pad)
            labels.append(label)
            numbers.append(label.split('_')[0])

        with open(f'{image_output}/labels.txt', 'w') as f:
            for label in labels:
                f.write('%s\n' % label)
        f.close()

        with open(f'{image_output}/numbers.txt', 'w') as f:
            for number in numbers:
                f.write('%s\n' % number)
        f.close()

        # Plot images and label
        for index in range(len(inputs_pad)):
            input_pad = inputs_pad[index]
            label = labels[index]
            street_segment = label.split('_')[1]
            number = label.split('_')[0]
            fname = f'{image_output}/index{index}.png'
            print(fname)
            save_inputs(input_pad, fname)
