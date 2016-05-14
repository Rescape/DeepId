import mxnet as mx
import logging
import argparse
from deepId_symbol import get_symbol 

parser = argparse.ArgumentParser(description='train an face classifer using DeepId model')
parser.add_argument('--save-model-prefix', type=str,
                    help='the prefix of the model to save')
parser.add_argument('--num-epochs', type=int, default=20,
                    help='the number of training epochs')
parser.add_argument('--batch-size', type=int, default=32,
                    help='the batch size')
# I only have one gpu(GTX970)
parser.add_argument('--gpus', type=str, default='0',
                    help='the gpus will be used, e.g "0,1,2,3"')
# Youtube_face should be 1595
parser.add_argument('--num-classes', type=int, default=1000,
                    help='the number of classes')

# I still don't know what kv_store's function here
parser.add_argument('--kv-store', type=str, default='local',
                    help='the kvstore type')

# Train set and Val set folder name(rec folder path and its name)
# Here mxnet will find the img data according to the content of rec
# First you should use im2rec tool to generate the rec file
parser.add_argument('--train-dataset', type=str, default="train.rec",
                    help='train dataset name')
parser.add_argument('--val-dataset', type=str, default="val.rec",
                    help="validation dataset name")


# About learnning rate 
parser.add_argument('--lr', type=float, default=.01,
                    help='the initial learning rate')
parser.add_argument('--lr-factor', type=float, default=1,
                    help='times the lr with a factor for every lr-factor-epoch epoch')
parser.add_argument('--lr-factor-epoch', type=float, default=1,
                    help='the number of epoch to factor the lr, could be .5')
# How to use?
parser.add_argument('--clip-gradient', type=float, default=5.,
                    help='clip min/max gradient to prevent extreme value')


# data dir
parser.add_argument('--data-dir', type=str, required=True,
                    help='the input data directory')
# need log to file?
parser.add_argument('--log-dir', type=str, default="/tmp/",
                    help='directory of the log file')
parser.add_argument('--train-dataset', type=str, default="train.rec",
                    help='train dataset name')
parser.add_argument('--val-dataset', type=str, default="val.rec",
                    help="validation dataset name")

# todo statistic about mean data

def get_iterator(args, kv):
    data_shape = (3, 47, 55)
    train = mx.io.ImageRecordIter(
        path_imgrec = os.path.join(args.data_dir, "train.bin"),
        # mean_r      = 123.68,
        # mean_g      = 116.779,
        # mean_b      = 103.939,
        data_shape  = data_shape,
        batch_size  = args.batch_size,
        rand_crop   = True,
        rand_mirror = True,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)

    val = mx.io.ImageRecordIter(
        path_imgrec = os.path.join(args.data_dir, "test.bin"),
        # mean_r      = 123.68,
        # mean_g      = 116.779,
        # mean_b      = 103.939,
        rand_crop   = False,
        rand_mirror = False,
        data_shape  = data_shape,
        batch_size  = args.batch_size,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)

    return (train, val)

import train_model
train_model.model.fit(args, net, get_iterator)

