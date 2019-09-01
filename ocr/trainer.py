try:
    from comet_ml import Experiment
    comet_loaded = True
except ImportError:
    comet_loaded = False

import os

from datetime import datetime
from easydict import EasyDict as edict
import numpy as np
import random
import skopt
from skopt.space import Real, Integer, Categorical
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from modules.film import FiLMGen
from test import validation
from SEVN_gym.envs.utils import convert_house_numbers
from load_ranges import get_random, get_sequence


def train(opt):
    '''
    Training function.

    Parameters
    ----------
    opt : argparse.Namespace
        Arguments to use for the experiment.

    Return
    ------
    neg_current_accuracy : float
        Return the negative of the best accuracy on validation set.

    '''
    # Checkpointing
    output_dir = './saved_models/{}'.format(opt.experiment_name)
    if opt.skopt:
        now = datetime.now
        output_dir += '/{}'.format(now().strftime('%Y-%m-%d-%H:%M:%S'))
    print('Output_dir: {}'.format(output_dir))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Seed
    print('Random Seed: ', opt.manualSeed)
    np.random.seed(opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True

    # GPU setting
    opt.num_gpu = torch.cuda.device_count()
    print('device count', opt.num_gpu)
    if opt.num_gpu > 1:
        print('------ Use multi-GPU setting ------')
        print('if you stuck too long time with multi-GPU setting,' +
              ' try to set --workers 0')
        # check multi-GPU issue https://github.com/clovaai/ocr/issues/1
        opt.workers = opt.workers * opt.num_gpu

    # Dataset preparation
    # Get a list of the selected dataset
    opt.select_data = opt.select_data.split('-')
    # Get a list of the ratio to use for the batch for each selected dataset
    opt.batch_ratio = opt.batch_ratio.split('-')
    train_dataset = Batch_Balanced_Dataset(opt)

    # Same as for training
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW,
                                      keep_ratio_with_pad=opt.PAD)
    valid_dataset = hierarchical_dataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=opt.batch_size,
        # 'True' to check training progress with validation function.
        shuffle=True,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid,
        pin_memory=True)
    print('-' * 80)

    # Model configuration
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3

    model = Model(opt)

    film_gen = FiLMGen(input_dim=200, emb_dim=opt.film_emb, cond_feat_size=opt.cond_feat_size, init_xavier=opt.init_xavier)
    if torch.cuda.is_available():
        film_gen.cuda()

    print('Model input parameters', opt.imgH, opt.imgW, opt.num_fiducial,
          opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length,
          opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    # Weight initialization
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue

    # Data parallel for multi-GPU
    model = torch.nn.DataParallel(model)
    if torch.cuda.is_available():
        model.cuda()
    model.train()
    film_gen.train()
    if opt.continue_model != '':
        print(f'loading pretrained model from {opt.continue_model}')
        model.load_state_dict(torch.load(opt.continue_model))
    print('Model:')
    print(model)

    # Setup loss
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True)
        if torch.cuda.is_available():
            criterion.cuda()
    else:
        # ignore [GO] token = ignore index 0
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        if torch.cuda.is_available():
            criterion.cuda()
    # Loss averager
    loss_avg = Averager()

    # Filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))

    film_filtered_params = []
    film_params_num = []
    for p in filter(lambda p: p.requires_grad, film_gen.parameters()):
        film_filtered_params.append(p)
        film_params_num.append(np.prod(p.size()))
    print(f'FiLM params num: {sum(film_params_num)}')
    print(f'Trainable params num : {sum(params_num)}')

    # Setup optimizer
    if opt.adam:
        optimizer = optim.Adam(
            filtered_parameters, lr=float(opt.lr), betas=(opt.beta1, 0.999))
        film_optimizer = optim.Adam(
            film_filtered_params, lr=float(opt.film_lr), betas=(opt.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(
            filtered_parameters, lr=float(opt.lr), rho=opt.rho, eps=opt.eps)
        film_optimizer = optim.Adadelta(
            film_filtered_params, lr=float(opt.film_lr), rho=opt.rho, eps=opt.eps)
    print('Optimizer:')
    print(optimizer)

    # Final options
    # Create comet experiment
    if comet_loaded and len(opt.comet) > 0 and not opt.no_comet:
        comet_credentials = opt.comet.split("/")
        experiment = Experiment(
            api_key=comet_credentials[2],
            project_name=comet_credentials[1],
            workspace=comet_credentials[0])
        for key, value in vars(opt).items():
            experiment.log_parameter(key, value)
    else:
        experiment = None

    # Print opt
    with open(f'{output_dir}/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)

    # Start training
    start_iter = 0
    # if opt.continue_model != '':
    #     start_iter = int(opt.continue_model.split('_')[-1].split('.')[0])
    #     print(f'continue to train, start_iter: {start_iter}')

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = 1e+6
    best_int_dist = 1e+10
    i = start_iter
    patience = opt.patience
    print(f'Opt.patience: {patience}')

    while(True):
        # Train part
        for p in model.parameters():
            p.requires_grad = True
        for p in film_gen.parameters():
            p.requires_grad = True

        # Get batch
        # image_tensors: transformed image, size [batch_size, c, h, w].
        # labels: list of labels, len batch size.
        image_tensors, labels = train_dataset.get_batch()

        # Labels preprocessing
        ids = labels
        # Labels are under the form 'doornumber_streetname'
        # Get the list of house number
        labels = [label.split('_')[0] for label in labels]
        if opt.sequential_cond:
            cond_house_numbers = [get_sequence(
                int(img_id.split('_')[0]),
                img_id.split('_')[1], opt.num_cond_hn) for img_id in ids]
        else:
            cond_house_numbers = [get_random(
                int(img_id.split('_')[0]),
                img_id.split('_')[1], opt.num_cond_hn) for img_id in ids]
        # Convert house number into vector of size 40.
        # Only house number with under 4 digits can be converted.
        assert all([len(str(n)) <= 4 for l in cond_house_numbers for n in l]),\
            'Maximum door numbers = 4'
        cond_house_numbers = [
            convert_house_numbers(n) for l in cond_house_numbers for n in l]
        cond_house_numbers = torch.FloatTensor(cond_house_numbers)
        # Reshape conditional text in a tensor of size [batch_size, 5*40]
        cond_text = cond_house_numbers.view(-1, opt.num_cond_hn * 40)

        image = image_tensors

        # labels is a list of house numbers
        # text correspond to the encoded house numbers
        # length correspond to sequence lenght + 1
        text, length = converter.encode(labels)
        batch_size = image.size(0)

        if torch.cuda.is_available():
            cond_text = cond_text.cuda()
            image = image.cuda()

        if 'CTC' in opt.Prediction:
            preds = model(image, text).log_softmax(2)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            preds = preds.permute(1, 0, 2)  # to use CTCLoss format
            cost = criterion(preds, text, preds_size, length)

        else:
            cond_params = None
            if opt.apply_film:
                cond_params = film_gen(cond_text)

            preds = model(image, text, cond_params)
            target = text[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.view(-1, preds.shape[-1]),
                             target.contiguous().view(-1))
        if opt.print_grad:
            try:
                print(f'model grad (sum): {sum([p.grad.sum() for p in model.parameters()])}')
                print(f'film_gen grad (sum): {sum([p.grad.sum() for p in film_gen.parameters()])}')
            except Exception:
                pass
        model.zero_grad()
        film_gen.zero_grad()
        cost.backward()

        # Gradient clipping with 5 (Default)
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()
        film_optimizer.step()
        print(f'cost: {cost.item()}')
        loss_avg.add(cost)

        # Validation part
        if i % opt.valInterval == 0:
            elapsed_time = time.time() - start_time
            print(f'[{i}/{opt.num_iter}] Loss: {loss_avg.val():0.5f}' +
                  f' elapsed_time: {elapsed_time:0.5f}')
            if experiment is not None:
                    experiment.log_metric("Time", elapsed_time, step=i)

            # For log file and comet
            with open(f'{output_dir}/log_train.txt', 'a') as log:
                log.write(f'[{i}/{opt.num_iter}] Loss: {loss_avg.val():0.5f}' +
                          f' elapsed_time: {elapsed_time:0.5f}\n')
                if experiment is not None:
                    experiment.log_metric('Loss', loss_avg.val(), step=i)
                loss_avg.reset()

                model.eval()
                film_gen.eval()
                valid_loss, current_accuracy, current_norm_ED, \
                    current_int_dist, preds, labels, infer_time, \
                    length_of_data = validation(
                        model, criterion, valid_loader,
                        converter, opt, film_gen=film_gen)
                model.train()
                film_gen.train()

                for pred, gt in zip(preds[:5], labels[:5]):
                    if 'Attn' in opt.Prediction:
                        pred = pred[:pred.find('[s]')]
                        gt = gt[:gt.find('[s]')]
                    print(f'{pred:20s}, gt: {gt:20s},   {str(pred == gt)}')
                    log.write(
                        f'{pred:20s}, gt: {gt:20s},   {str(pred == gt)}\n')

                valid_log = f'[{i}/{opt.num_iter}]'
                valid_log += f' valid loss: {valid_loss:0.5f}'
                valid_log += f' accuracy: {current_accuracy:0.3f},'
                valid_log += f' norm_ED: {current_norm_ED:0.2f},'
                valid_log += f' int_dist: {current_int_dist:0.2f}'
                print(valid_log)
                log.write(valid_log + '\n')
                if experiment is not None:
                    experiment.log_metric("Valid Loss", valid_loss, step=i)
                    experiment.log_metric("Accuracy", current_accuracy, step=i)
                    experiment.log_metric("Norm ED", current_norm_ED, step=i)
                    experiment.log_metric("Int Distance",
                                          current_int_dist, step=i)

                # Keep best accuracy model
                if current_accuracy > best_accuracy:
                    patience = opt.patience
                    best_accuracy = current_accuracy
                    torch.save(model.state_dict(),
                               f'{output_dir}/best_accuracy.pth')
                    torch.save(film_gen.state_dict(),
                               f'{output_dir}/best_accuracy_film_gen.pth')
                else:
                    patience -= 1
                    if patience == 0:
                        print("Early stopping.")

                        # Writeout the final results, input images, predicted text, conditioning text (Extracted from 10x4, best accuracy)
                        with open(f'{output_dir}/log_final.txt', 'a') as log_final:
                            log_final.write(f'best_accuracy: {best_accuracy:0.3f},' + \
                                f' best_norm_ED: {best_norm_ED:0.2f},' + f' best_int_dist: {best_int_dist:0.2f} \n')

                            model.load_state_dict(torch.load(f'{output_dir}/best_accuracy.pth'))
                            valid_loss, current_accuracy, current_norm_ED, \
                                current_int_dist, preds, labels, infer_time, \
                                length_of_data = validation(
                                    model, criterion, valid_loader,
                                    converter, opt, film_gen=film_gen,
                                    output_dir=output_dir, final_eval=True)
                        break

                if current_norm_ED < best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(model.state_dict(),
                               f'{output_dir}/best_norm_ED.pth')
                best_model_log = f'best_accuracy: {best_accuracy:0.3f},' + \
                    f' best_norm_ED: {best_norm_ED:0.2f}'
                print(best_model_log)
                log.write(best_model_log + '\n')

                if current_int_dist < best_int_dist:
                    best_int_dist = current_int_dist

                if experiment is not None:
                    experiment.log_metric('Best Accuracy',
                                          best_accuracy, step=i)
                    experiment.log_metric('Best Norm ED',
                                          best_norm_ED, step=i)

        # Save model per 1e+5 iter.
        if (i + 1) % 1e+5 == 0:
            torch.save(
                model.state_dict(), f'{output_dir}/iter_{i+1}.pth')

        if i == opt.num_iter:
            print('end the training')
            break
        i += 1
    neg_current_accuracy = -current_accuracy
    return neg_current_accuracy


def train_skopt(args, skopt_n_calls=100, skopt_random_state=0):
    '''
    Do a random search optimization.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments to use for the experiment.
    n_calls : int
        Number or random search to perform. Default 100.
    random_state : int
        Seed to use for the skopt function. Default 0.

    '''
    # Find hyperparameters to optimize
    SPACE = []
    STATIC_PARAMS = {}
    for arg in vars(args):
        value = getattr(args, arg)
        if isinstance(value, (Real, Integer, Categorical)):
            SPACE.append(value)
        else:
            STATIC_PARAMS[arg] = value

    @skopt.utils.use_named_args(SPACE)
    def objective(**params):
        '''
        Function to optimize. See train function for more details.
        '''
        all_params = edict({**params, **STATIC_PARAMS})
        return train(all_params)

    # Run hyperparameters search
    results = skopt.dummy_minimize(objective, SPACE,
                                   n_calls=skopt_n_calls,
                                   random_state=skopt_random_state)
    # Save skopt results
    skopt_fname = f'./saved_models/{args.experiment_name}/skopt_results.pkl'
    results.specs['args'].pop('func', None)
    skopt.dump(results, skopt_fname)
