import os
import sys

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '/utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module
from loss import *


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 9))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    sem_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl, sem_pl


def get_model(point_cloud, is_training, num_class, bn_decay=None):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud[:, :, :3]
    l0_points = point_cloud[:, :, 3:]
    end_points['l0_xyz'] = l0_xyz

    # Layer 1
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=1024, radius=0.1, nsample=32,
                                                       mlp=[32, 32, 64], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=256, radius=0.2, nsample=32,
                                                       mlp=[64, 64, 128], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=64, radius=0.4, nsample=32,
                                                       mlp=[128, 128, 256], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer3')
    l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=16, radius=0.8, nsample=32,
                                                       mlp=[256, 256, 512], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer4')

    # Feature Propagation layers
    l3_points_sem = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256, 256], is_training, bn_decay,
                                       scope='sem_fa_layer1')
    l2_points_sem = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points_sem, [256, 256], is_training, bn_decay,
                                       scope='sem_fa_layer2')
    l1_points_sem = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points_sem, [256, 128], is_training, bn_decay,
                                       scope='sem_fa_layer3')
    l0_points_sem = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points_sem, [128, 128, 128], is_training, bn_decay,
                                       scope='sem_fa_layer4')

    # FC layers
    net_sem = tf_util.conv1d(l0_points_sem, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='sem_fc1',
                             bn_decay=bn_decay)
    net_sem_cache = tf_util.conv1d(net_sem, 128, 1, padding='VALID', bn=True, is_training=is_training,
                                   scope='sem_cache', bn_decay=bn_decay)

    # ins

    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    point_cloud_transformed = tf.matmul(point_cloud, transform)
    input_image = tf.expand_dims(point_cloud_transformed, -1)

    spatial_net = tf_util.conv2d(input_image, 64, [1, 3], padding='VALID', stride=[1,1],bn=True, is_training=is_training,
                                scope='conv1', bn_decay=bn_decay)
    spatial_net = tf_util.conv2d(spatial_net, 64, [1,1],padding='VALID', stride=[1,1],bn=True, is_training=is_training,
                                scope='conv2', bn_decay=bn_decay)

    with tf.variable_scope('transform_net2') as sc:
        transform = feature_transform_net(spatial_net, is_training, bn_decay, K=64)
    net_transformed = tf.matmul(tf.squeeze(spatial_net, axis=[2]), transform)
    net_transformed = tf.expand_dims(net_transformed, [2])


    net_ins = tf.concat([net_sem_cache, net_transformed], axis=3)
    net_ins = tf_util.conv2d(net_ins, 192, [1, 1], padding='VALID', stride=[1,1], activation_fn=None, is_training=is_training,
                             scope='net_ins1', bn_decay=bn_decay)
    net_ins = tf_util.conv2d(net_ins, 128, [1, 1], padding='VALID', stride=[1, 1], activation_fn=None,
                             is_training=is_training,
                             scope='net_ins1', bn_decay=bn_decay)
    net_ins = tf_util.conv2d(net_ins, 128, [1, 1], padding='VALID', stride=[1, 1], activation_fn=None,
                             is_training=is_training,
                             scope='net_ins2', bn_decay=bn_decay)
    net_ins = tf.squeeze(net_ins, axis=2)

    k = 30
    adj_matrix = tf_util.pairwise_distance_cosine(net_ins)
    nn_idx = tf_util.knn_thres(adj_matrix, k=k)
    nn_idx = tf.stop_gradient(nn_idx)
    net_sem = tf_util.get_local_feature(net_sem, nn_idx=nn_idx, k=k)  # [b, n, k, c]
    net_sem = tf.reduce_max(net_sem, axis=-2, keep_dims=False)

    net_sem = tf_util.dropout(net_sem, keep_prob=0.5, is_training=is_training, scope='sem_dp1')
    net_sem = tf_util.conv1d(net_sem, num_class, 1, padding='VALID', activation_fn=None, scope='sem_fc4')

    print("net_sem: " + str(tf.shape(net_sem)))
    print("net_ins: " + str(tf.shape(net_ins)))

    return net_sem, net_ins


def get_loss(pred, ins_label, pred_sem, sem_label, alpha=0.5):
    """
    :param pred:  net_ins B*N*E
    :param ins_label: B*N
    :param pred_sem: B*N*13
    :param sem_label: B*N
    :param alpha: hyperparameter in cos_loss
    :return: total loss and other loss
    """
    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=sem_label, logits=pred_sem)
    tf.summary.scalar('classify loss', classify_loss)

    feature_dim = pred.get_shape().as_list()[-1]
    num_point = pred.get_shape().as_list()[1]

    cos_loss, same_loss, neq_loss = cosine_loss(pred, ins_label, alpha)

    loss = classify_loss + cos_loss

    tf.add_to_collection('losses', loss)

    return loss, classify_loss, cos_loss, same_loss, neq_loss


if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32, 2048, 3))
        net, _ = get_model(inputs, tf.constant(True), 10)
        print(net)

