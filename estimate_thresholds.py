import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
from scipy import stats

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
print(ROOT_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

import provider
import tf_util
from model import *
from test_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
parser.add_argument('--model_path', type=str, default='log/model.ckpt', help='Path of model')
parser.add_argument('--test_area', type=int, default=5, help='The areas expcept this one will be used to estimate mean thresholds')
FLAGS = parser.parse_args()

NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu
MODEL_PATH = FLAGS.model_path
TEST_AREA = FLAGS.test_area
LOG_DIR = 'log{}'.format(TEST_AREA)
ESTIMATE_FILE_PATH = "data/train_hdf5_file_list_woArea{}.txt".format(TEST_AREA)

BATCH_SIZE = 1
MAX_NUM_POINT = 4096
NUM_CLASSES = 13


def estimate_mean_thresholds():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            pointclouds_pl, labels_pl, sem_labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            # Get model
            pred_sem, pred_ins = get_model(pointclouds_pl, is_training_pl, NUM_CLASSES)

            loader = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True
        sess = tf.Session(config=config)

        is_training = False

        # Restore all the variables
        loader.restore(sess, MODEL_PATH)
        print("Model restored")
        train_file_list = provider.getDataFiles(ESTIMATE_FILE_PATH)

        ths = np.zeros(NUM_CLASSES)
        cnt = np.zeros(NUM_CLASSES)
        ths_ = np.zeros(NUM_CLASSES)

        for shape_idx in range(len(train_file_list)):
            h5_filename = train_file_list[shape_idx]
            print('Loading file ' + h5_filename)
            cur_data, cur_group, _, cur_sem = provider.loadDataFile_with_groupseglabel_stanfordindoor(h5_filename)
            num_data = cur_data.shape[0]
            for j in range(num_data):
                print("Processing: Shape [%d] Block[%d]" % (shape_idx, j))
                pts = cur_data[j, ...]
                seg = cur_sem[j, ...]
                ins = cur_group[j, ...]
                feed_dict = {
                    pointclouds_pl: np.expand_dims(pts, 0),
                    is_training_pl: is_training
                }
                pred_ins_val = sess.run([pred_ins], feed_dict=feed_dict)
                pred_ins_val = np.squeeze(pred_ins_val, axis=0)
                dis_mat = np.expand_dims(pred_ins_val, axis=1) - np.expand_dims(pred_ins_val, axis=0)
                dis_mat = np.linalg.norm(dis_mat, ord=1, axis=2)
                ths, ths_, cnt = Get_Ths(dis_mat, seg, ins, ths, ths_, cnt)
                print(cnt)
        ths = [ths[i] / cnt[i] if cnt[i] != 0 else 0.2 for i in range(len(cnt))]
        print(ths)
        np.savetxt(os.path.join(LOG_DIR, 'mean_thresholds.txt'), ths)


if __name__ == "__main__":
    estimate_mean_thresholds()