import os

import argparse
import ruamel.yaml as yaml
from skopt.space import Real, Integer, Categorical

from trainer import train, train_skopt


def parse_args():
    parser = argparse.ArgumentParser()
    """ Experiment hyperparameters """
    parser.add_argument('--train_data',
                        type=str, default='data/train',
                        help='Path to training dataset.')
    parser.add_argument('--valid_data',
                        type=str, default='data/valid',
                        help='Path to validation dataset.')
    parser.add_argument('--experiment_name',
                        type=str, default='',
                        help='''Optional custom experiment name.
                              If not set experiment name will be
                              "{args.Transformation}-{args.FeatureExtraction}
                               -{args.SequenceModeling}-{args.Prediction}
                               -Seed{args.manualSeed}".
                               Logs and models will be stored in
                               "./saved_models/{opt.experiment_name}".
                              ''')
    parser.add_argument('--skopt',
                        action='store_true',
                        help='Apply hyperparameter random search with skopt.')
    parser.add_argument('--skopt_n_calls',
                        type=int, default=100,
                        help='Number or random search to perform.')
    parser.add_argument('--skopt_random_state',
                        type=int, default=0,
                        help='Seed to use for the skopt function.')
    parser.add_argument('--skopt_cfg',
                        type=str, default='',
                        help='''Path to the file containing the hyperparameters
                             to optimize.''')
    parser.add_argument('--manualSeed',
                        type=int, default=1111,
                        help='For random seed setting.')
    parser.add_argument('--workers',
                        type=int, default=4,
                        help='Number of data loading workers.')
    parser.add_argument('--batch_size',
                        type=int, default=192,
                        help='Input batch size.')
    parser.add_argument('--num_iter',
                        type=int, default=300000,
                        help='Number of iterations to train for.')
    parser.add_argument('--valInterval',
                        type=int, default=10,
                        help='Interval between each validation.')
    parser.add_argument('--continue_model',
                        type=str, default='',
                        help='Path to model to continue training.')
    parser.add_argument('--adam',
                        action='store_true',
                        help='Whether to use adam (default is Adadelta).')
    parser.add_argument('--lr',
                        type=str, default=1,
                        help='Learning rate, default=1.0 for Adadelta.')
    parser.add_argument('--film_lr',
                        type=str, default=1,
                        help='Film learning rate, default=1.0 for Adadelta.')
    parser.add_argument('--beta1',
                        type=float, default=0.9,
                        help='beta1 for adam. default=0.9')
    parser.add_argument('--rho',
                        type=float, default=0.95,
                        help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps',
                        type=float, default=1e-8,
                        help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip',
                        type=float, default=5,
                        help='Gradient clipping value. default=5')
    parser.add_argument('--patience',
                        type=float, default=10,
                        help='Patience value for early stopping.')
    """ Data processing """
    parser.add_argument('--select_data',
                        type=str, default='data',
                        help='''Select training data (default is data).''')
    parser.add_argument('--batch_ratio',
                        type=str, default='1.0',
                        help='''Assign ratio for each selected data in the
                                batch.''')
    parser.add_argument('--total_data_usage_ratio',
                        type=str, default='1.0',
                        help='''Total data usage ratio, this ratio is
                                multiplied to total number of data.''')
    parser.add_argument('--batch_max_length',
                        type=int, default=25,
                        help='Maximum-label-length.')
    parser.add_argument('--imgH',
                        type=int, default=32,
                        help='The height of the input image.')
    parser.add_argument('--imgW',
                        type=int, default=100,
                        help='The width of the input image.')
    parser.add_argument('--rgb',
                        action='store_true',
                        help='Use rgb input.')
    parser.add_argument('--character',
                        type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz',
                        help='character label')
    parser.add_argument('--sensitive',
                        action='store_true',
                        help='For sensitive character mode.')
    parser.add_argument('--PAD',
                        action='store_true',
                        help='''Whether to keep ratio then pad for
                                image resize.''')
    """ Model Architecture """
    parser.add_argument('--Transformation',
                        type=str, default='TPS',
                        help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction',
                        type=str, default='ResNet',
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling',
                        type=str, default='BiLSTM',
                        help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction',
                        type=str, default='Attn',
                        help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial',
                        type=int, default=20,
                        help='Number of fiducial points of TPS-STN.')
    parser.add_argument('--input_channel',
                        type=int, default=1,
                        help='''The number of input channel of Feature
                                extractor.''')
    parser.add_argument('--output_channel',
                        type=int, default=512,
                        help='''The number of output channel of Feature
                                extractor.''')
    parser.add_argument('--hidden_size',
                        type=int, default=256,
                        help='The size of the LSTM hidden state.')
    parser.add_argument('--comet',
                        type=str,
                        default='mweiss17/navi-str/UcVgpp0wPaprHG4w8MFVMgq7j',
                        help='Comet logging info.')
    parser.add_argument('--no_comet',
                        action='store_true',
                        help='Dont use comet.')
    parser.add_argument('--ed_condition',
                        action='store_true',
                        help='???')
    parser.add_argument('--apply_film',
                        action='store_true',
                        help='Apply film to the feature extractor model.')
    parser.add_argument('--freeze_main',
                        action='store_true',
                        help='Freeze the main models network.')
    parser.add_argument('--film_emb',
                        type=int, default=256,
                        help='Define the hidden size for the film network.')
    parser.add_argument('--film_layers',
                        type=int, default=1,
                        help='Define the number of hidden layers for the film network.')
    parser.add_argument('--film_transformer',
                        action='store_true',
                        help='Define the network architecture for film generator network.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Load arguments
    args = parse_args()

    if not args.experiment_name:
        experiment_name = f'{args.Transformation}-{args.FeatureExtraction}'
        experiment_name += f'-{args.SequenceModeling}-{args.Prediction}'
        experiment_name += f'-Seed{args.manualSeed}'
        args.experiment_name = experiment_name

    if args.skopt:
        # Load skopt config file
        if not args.skopt_cfg:
            raise Exception('No skopt config file specified.')

        with open(args.skopt_cfg, 'r') as f:
            skopt_cfg = yaml.safe_load(f)

        # Merge skopt_cfg into args
        intersection = set(vars(args)).intersection(set(skopt_cfg))
        print('Hyperparameters to optimize: {}'.format(list(intersection)))
        for arg in intersection:
            skopt_value = eval(skopt_cfg[arg])
            if not isinstance(skopt_value, (Real, Integer, Categorical)):
                raise Exception(f'{skopt_value} not an skopt space.')
            else:
                skopt_value.name = arg
            vars(args)[arg] = skopt_value

        # Train with hyperparameters tuning with skopt
        train_skopt(args, args.skopt_n_calls, args.skopt_random_state)
    else:
        # Train model
        train(args)
