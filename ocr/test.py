import os
import time
import string
import argparse
import random
import re

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn as nn
import torchvision
import numpy as np
import pandas as pd
from nltk.metrics.distance import edit_distance

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate
from model import Model
from modules.film import FiLMGen
from SEVN_gym.envs.utils import convert_street_name, convert_house_numbers
from load_ranges import get_random, get_sequence


def int_distance(num, gt_num):
    num = re.sub("[^0-9]", "", num)
    try:
        num = int(num)
    except:
        num = 0
    distance = np.abs(num - int(gt_num))
    return distance

def benchmark_all_eval(model, criterion, converter, opt, calculate_infer_time=False):
    """ evaluation with 10 benchmark evaluation datasets """
    # The evaluation datasets, dataset order is same with Table 1 in our paper.
    eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_860', 'IC03_867', 'IC13_857',
                      'IC13_1015', 'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80']

    if calculate_infer_time:
        evaluation_batch_size = 1  # batch_size should be 1 to calculate the GPU inference time per image.
    else:
        evaluation_batch_size = opt.batch_size

    list_accuracy = []
    total_forward_time = 0
    total_evaluation_data_number = 0
    total_correct_number = 0
    print('-' * 80)
    for eval_data in eval_data_list:
        eval_data_path = os.path.join(opt.eval_data, eval_data)
        AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        eval_data = hierarchical_dataset(root=eval_data_path, opt=opt)
        evaluation_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=evaluation_batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_evaluation, pin_memory=True)

        _, accuracy_by_best_model, norm_ED_by_best_model, int_dist_by_best_model, _, _, infer_time, length_of_data = validation(
            model, criterion, evaluation_loader, converter, opt)
        list_accuracy.append(f'{accuracy_by_best_model:0.3f}')
        total_forward_time += infer_time
        total_evaluation_data_number += len(eval_data)
        total_correct_number += accuracy_by_best_model * length_of_data
        print('Acc %0.3f\t normalized_ED %0.3f\t int_distance %0.3f' % (accuracy_by_best_model, norm_ED_by_best_model, int_dist_by_best_model))
        print('-' * 80)

    averaged_forward_time = total_forward_time / total_evaluation_data_number * 1000
    total_accuracy = total_correct_number / total_evaluation_data_number
    params_num = sum([np.prod(p.size()) for p in model.parameters()])

    evaluation_log = 'accuracy: '
    for name, accuracy in zip(eval_data_list, list_accuracy):
        evaluation_log += f'{name}: {accuracy}\t'
    evaluation_log += f'total_accuracy: {total_accuracy:0.3f}\t'
    evaluation_log += f'averaged_infer_time: {averaged_forward_time:0.3f}\t# parameters: {params_num/1e6:0.3f}'
    print(evaluation_log)
    with open(f'./result/{opt.experiment_name}/log_all_evaluation.txt', 'a') as log:
        log.write(evaluation_log + '\n')

    return None


def validation(model, criterion, evaluation_loader, converter, opt, eval_data=None, film_gen=None):
    """ validation or evaluation """
    for p in model.parameters():
        p.requires_grad = False

    n_correct = 0
    norm_ED = 0
    int_dist = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()

    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        ids = labels
        labels = [label.split("_")[0] for label in labels]

        with torch.no_grad():
            image = image_tensors.cuda()
            # For max length prediction
            length_for_pred = torch.cuda.IntTensor([opt.batch_max_length] * batch_size)
            text_for_pred = torch.cuda.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0)

            text_for_loss, length_for_loss = converter.encode(labels)

        start_time = time.time()
        if 'CTC' in opt.Prediction:
            preds = model(image, text_for_pred).log_softmax(2)
            forward_time = time.time() - start_time

            # Calculate evaluation loss for CTC deocder.
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            preds = preds.permute(1, 0, 2)  # to use CTCloss format
            cost = criterion(preds, text_for_loss, preds_size, length_for_loss)

            # Select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
            preds_str = converter.decode(preds_index.data, preds_size.data)

        else:
            num_hn = 5
            num_ed_text = 20
            cond_house_numbers = [get_random(int(img_id.split("_")[0]), img_id.split("_")[1], num_hn) for img_id in ids]
            ed_text = [get_random(int(img_id.split("_")[0]), img_id.split("_")[1], num_ed_text) for img_id in ids]
            cond_house_numbers = [convert_house_numbers(n) for l in cond_house_numbers for n in l]
            cond_house_numbers = torch.FloatTensor(cond_house_numbers)
            cond_text = cond_house_numbers.view(-1, num_hn * 40).cuda()
            cond_params = []
            if opt.apply_film:
                cond_params.append(film_gen(cond_text))

            preds = model(image, text_for_pred, cond_params, is_train=False)
            forward_time = time.time() - start_time

            preds = preds[:, :text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

        infer_time += forward_time
        valid_loss_avg.add(cost)

        # calculate accuracy.
        j = 0
        for pred, gt, text in zip(preds_str, labels, ed_text):
            if 'Attn' in opt.Prediction:
                pred = pred[:pred.find('[s]')]  # prune after "end of sentence" token ([s])
                gt = gt[:gt.find('[s]')]

            if opt.ed_condition:
                text = [str(num) for num in text]
                if gt not in text:
                    text[0] = gt
                    random.shuffle(text)

                distances = {}
                for word in text:
                    d = edit_distance(word, pred)
                    distances[word] = d
                pred = min(distances, key=distances.get)

            if pred == gt:
                n_correct += 1
                if eval_data:
                    eval_data[i * batch_size + j][0].save(f"./result/{opt.experiment_name}/correct/sample_{i * batch_size + j}_pred_{pred}.png")
            else:
                if eval_data:
                    eval_data[i * batch_size + j][0].save(f"./result/{opt.experiment_name}/incorrect/sample_{i * batch_size + j}_pred_{pred}.png")

            norm_ED += edit_distance(pred, gt) / len(gt)
            int_dist += int_distance(pred, gt)
            j += 1

    accuracy = n_correct / float(length_of_data) * 100

    return valid_loss_avg.val(), accuracy, norm_ED, int_dist, preds_str, labels, infer_time, length_of_data


def test(opt):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).cuda()

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model))
    film_gen = FiLMGen(input_dim=200, emb_dim=1000, cond_feat_size=18944).cuda()

    if opt.apply_film:
        film_gen.load_state_dict(torch.load(f'./{"/".join(opt.saved_model.split("/")[:-1])}/best_accuracy_film_gen.pth'))
    # print(model)

    """ keep evaluation model and result logs """
    os.makedirs(f'./result/{opt.experiment_name}', exist_ok=True)
    os.makedirs(f'./result/{opt.experiment_name}/correct', exist_ok=True)
    os.makedirs(f'./result/{opt.experiment_name}/incorrect', exist_ok=True)
    os.system(f'cp {opt.saved_model} ./result/{opt.experiment_name}/')

    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).cuda()  # ignore [GO] token = ignore index 0

    """ evaluation """
    model.eval()
    if opt.benchmark_all_eval:  # evaluation with 10 benchmark evaluation datasets
        benchmark_all_eval(model, criterion, converter, opt)
    else:
        AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        eval_data = hierarchical_dataset(root=opt.eval_data, opt=opt)
        evaluation_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=opt.batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_evaluation, pin_memory=True)

        _, accuracy_by_best_model, norm_ED, int_dist, _, _, _, _ = validation(
            model, criterion, evaluation_loader, converter, opt, eval_data, film_gen=film_gen)

        print(accuracy_by_best_model)
        with open('./result/{0}/log_evaluation.txt'.format(opt.experiment_name), 'w') as log:
            log.write(str(accuracy_by_best_model) + '\n')
            log.write(str(norm_ED) + '\n')
            log.write(str(int_dist) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_data', required=True, help='path to evaluation dataset')
    parser.add_argument('--experiment_name', required=True, help='Experiment name')
    parser.add_argument('--benchmark_all_eval', action='store_true', help='evaluate 10 benchmark evaluation datasets')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    parser.add_argument('--ed_condition', action="store_true", help='Matching')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--apply_film', action="store_true", help='Apply film to the')

    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    random.seed(opt.seed)

    test(opt)
