import argparse


def parse_args_base(parser):
    parser.add_argument('--n', help='network depth', type=int, default=3)
    parser.add_argument('--batch', help='batch size', type=int, default=128)
    parser.add_argument('--decay_milestones', nargs='+', type=int, help='Epochs at which to decay learning rate', default=[50, 100])
    parser.add_argument('--dataset_dir', default='./dataset')
    return parser


def parse_args_train():
    parser = argparse.ArgumentParser()
    parser = parse_args_base(parser)
    parser.add_argument('--checkpoint_dir', default='./checkpoint')
    parser.add_argument('--print_freq', help='print loss freq', type=int, default=10)
    parser.add_argument('--save_params_freq', help='parameters saving freq', type=int, default=500)
    parser.add_argument('--lr', help='initial learning rate', type=float, default=0.1)
    parser.add_argument('--momentum', help='optimizer momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', help='optimizer weight decay (L2 reg.)', type=float, default=0.0005)
    parser.add_argument('--decay_lr_1', help='iteration at which lr decays 1st', type=int, default=32000)
    parser.add_argument('--decay_lr_2', help='iteration at which lr decays 2nd', type=int, default=48000)
    parser.add_argument('--lr_decay_rate', help='lr *= lr_decay_rate at decay_lr_i-th iteration', type=float, default=0.1)
    parser.add_argument('--n_iter', help='learning iterations', type=int, default=80000)
    # Add an argument to select the optimizer
    parser.add_argument('--optimizer', help='optimizer to use (sgd, momentum, adam)', type=str, default='sgd', choices=['sgd', 'momentum', 'adam'])
    args = parser.parse_args()

    return args


def parse_args_test():
    parser = argparse.ArgumentParser()
    parser = parse_args_base(parser)
    parser.add_argument('--params_path', help='path to saved model weights')
    args = parser.parse_args()

    return args
