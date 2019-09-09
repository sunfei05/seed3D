# -*- coding: utf-8 -*-
import os
import sys
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))


def discriminative_loss_single(prediction, correct_label, feature_dim,
                               delta_v, delta_d, param_var, param_dist, param_reg):
    ''' Discriminative loss for a single prediction/label pair.
    :param prediction: inference of network N*E
    :param correct_label: instance label    N
    :feature_dim: feature dimension of prediction
    :param label_shape: shape of label
    :param delta_v: cutoff variance distance
    :param delta_d: cutoff cluster distances
    :param param_var: weight for intra cluster variance
    :param param_dist: weight for inter cluster distances
    :param param_reg: weight regularization
    '''

    ### Reshape so pixels are aligned along a vector
    # correct_label = tf.reshape(correct_label, [label_shape[1] * label_shape[0]])
    reshaped_pred = tf.reshape(prediction, [-1, feature_dim])

    ### Count instances
    """
    tf.unique_with_counts:
    eg.:
    x:[1,1,2,4,4,4,7,8,8]
    y,idx,count = tf.unique_with_counts(x)
    y: [1,2,4,7,8]
    idx: [0,0,1,2,2,2,3,4,4]
    count: [2,1,3,1,2]
    """

    unique_labels, unique_id, counts = tf.unique_with_counts(correct_label)
    # counts中记录了每个instance中的点数
    counts = tf.cast(counts, tf.float32)
    num_instances = tf.size(unique_labels)

    """
    tf.unsorted_segment_sum(data, idx, num_segments):
    idx相同的data相加并将最终结果放置在output[idx]上
    eg:
    data:[[1 2 3]
          [4 5 6]
          [7 8 9]]
    idx:[0 1 0]
    num_seg: 2
    y = tf.unsorted_segment_sum(data, idx, num_seg)
    y:
    [[8 10 12]
     [4 5 6]]
    """
    segmented_sum = tf.unsorted_segment_sum(reshaped_pred, unique_id, num_instances)
    # segmented_sum记录了每个实例在嵌入空间上的向量和

    mu = tf.div(segmented_sum, tf.reshape(counts, (-1, 1)))
    # mu得到了每个实例的中心

    mu_expand = tf.gather(mu, unique_id)
    # mu_expand得到了每个点对应的实例中心
    # mu, mu_expand = instance_center(prediction, correct_label, feature_dim, delta_v)
    ### Calculate l_var
    # distance = tf.norm(tf.subtract(mu_expand, reshaped_pred), axis=1)
    # tmp_distance = tf.subtract(reshaped_pred, mu_expand)
    tmp_distance = reshaped_pred - mu_expand
    # tmp_distance.shape: N*E
    # tf.norm: 求取范数 ord=1即求取1范数，即绝对值之和
    distance = tf.norm(tmp_distance, ord=1, axis=1)

    distance = tf.subtract(distance, delta_v)
    distance = tf.clip_by_value(distance, 0., distance)  # 减去负数部分
    distance = tf.square(distance)

    l_var = tf.unsorted_segment_sum(distance, unique_id, num_instances)
    l_var = tf.div(l_var, counts)
    l_var = tf.reduce_sum(l_var)
    l_var = tf.divide(l_var, tf.cast(num_instances, tf.float32))

    ### Calculate l_dist

    # Get distance for each pair of clusters like this:
    #   mu_1 - mu_1
    #   mu_2 - mu_1
    #   mu_3 - mu_1
    #   mu_1 - mu_2
    #   mu_2 - mu_2
    #   mu_3 - mu_2
    #   mu_1 - mu_3
    #   mu_2 - mu_3
    #   mu_3 - mu_3
    # tf.tile：[num_instances, 1] 第一维复制num_instances遍，第二维复制1遍。
    mu_interleaved_rep = tf.tile(mu, [num_instances, 1])
    mu_band_rep = tf.tile(mu, [1, num_instances])
    mu_band_rep = tf.reshape(mu_band_rep, (num_instances * num_instances, feature_dim))

    mu_diff = tf.subtract(mu_band_rep, mu_interleaved_rep)

    # Filter out zeros from same cluster subtraction
    # eye.shape: num_instances * num_instances的单位矩阵
    eye = tf.eye(num_instances)
    zero = tf.zeros(1, dtype=tf.float32)
    diff_cluster_mask = tf.equal(eye, zero)
    diff_cluster_mask = tf.reshape(diff_cluster_mask, [-1])
    mu_diff_bool = tf.boolean_mask(mu_diff, diff_cluster_mask)

    mu_norm = tf.norm(mu_diff_bool, ord=1, axis=1)
    mu_norm = tf.subtract(2. * delta_d, mu_norm)
    mu_norm = tf.clip_by_value(mu_norm, 0., mu_norm)
    mu_norm = tf.square(mu_norm)

    l_dist = tf.reduce_mean(mu_norm)

    def rt_0(): return 0.

    def rt_l_dist(): return l_dist

    # 实例个数等于1时，l_dist部分的损失为0
    l_dist = tf.cond(tf.equal(1, num_instances), rt_0, rt_l_dist)

    ### Calculate l_reg
    l_reg = tf.reduce_mean(tf.norm(mu, ord=1, axis=1))

    param_scale = 1.
    l_var = param_var * l_var
    l_dist = param_dist * l_dist
    l_reg = param_reg * l_reg

    loss = param_scale * (l_var + l_dist + l_reg)

    return loss, l_var, l_dist, l_reg


def discriminative_loss(prediction, correct_label, feature_dim,
                        delta_v, delta_d, param_var, param_dist, param_reg):
    '''
    Iterate over a batch of prediction/label and cumulate loss
    prediction: B*N*E 预测出的每个点在特征空间中的向量
    :return: discriminative loss and its three components
    对于每个num_point求取中心，while_loop
    '''

    def cond(label, batch, out_loss, out_var, out_dist, out_reg, i):
        return tf.less(i, tf.shape(batch)[0])

    def body(label, batch, out_loss, out_var, out_dist, out_reg, i):
        disc_loss, l_var, l_dist, l_reg = discriminative_loss_single(prediction[i], correct_label[i], feature_dim,
                                                                     delta_v, delta_d, param_var, param_dist, param_reg)

        out_loss = out_loss.write(i, disc_loss)
        out_var = out_var.write(i, l_var)
        out_dist = out_dist.write(i, l_dist)
        out_reg = out_reg.write(i, l_reg)

        return label, batch, out_loss, out_var, out_dist, out_reg, i + 1

    # TensorArray is a data structure that support dynamic writing
    output_ta_loss = tf.TensorArray(dtype=tf.float32,
                                    size=0,
                                    dynamic_size=True)
    output_ta_var = tf.TensorArray(dtype=tf.float32,
                                   size=0,
                                   dynamic_size=True)
    output_ta_dist = tf.TensorArray(dtype=tf.float32,
                                    size=0,
                                    dynamic_size=True)
    output_ta_reg = tf.TensorArray(dtype=tf.float32,
                                   size=0,
                                   dynamic_size=True)

    """
    tf.while_loop: final_state = tf.while_loop(cond, loop_body, init_state)
    state = init_state
    while(cond(state)):
        state = loop_body(state)
    return state
    """
    _, _, out_loss_op, out_var_op, out_dist_op, out_reg_op, _ = tf.while_loop(cond, body, [correct_label,
                                                                                           prediction,
                                                                                           output_ta_loss,
                                                                                           output_ta_var,
                                                                                           output_ta_dist,
                                                                                           output_ta_reg,
                                                                                           0])
    out_loss_op = out_loss_op.stack()
    out_var_op = out_var_op.stack()
    out_dist_op = out_dist_op.stack()
    out_reg_op = out_reg_op.stack()

    disc_loss = tf.reduce_mean(out_loss_op)
    l_var = tf.reduce_mean(out_var_op)
    l_dist = tf.reduce_mean(out_dist_op)
    l_reg = tf.reduce_mean(out_reg_op)

    return disc_loss, l_var, l_dist, l_reg


def discriminative_loss_single_multicate(sem_label, prediction, correct_label, feature_dim,
                                         delta_v, delta_d, param_var, param_dist, param_reg):
    ''' Discriminative loss for a single prediction/label pair.
    :param sem_label: semantic label
    :param prediction: inference of network
    :param correct_label: instance label
    :feature_dim: feature dimension of prediction
    :param label_shape: shape of label
    :param delta_v: cutoff variance distance
    :param delta_d: curoff cluster distance
    :param param_var: weight for intra cluster variance
    :param param_dist: weight for inter cluster distances
    :param param_reg: weight regularization
    '''
    unique_sem_label, unique_id, counts = tf.unique_with_counts(sem_label)
    num_sems = tf.size(unique_sem_label)

    def cond(i, ns, unique_id, pred, ins_label, out_loss, out_var, out_dist, out_reg):
        return tf.less(i, num_sems)

    def body(i, ns, unique_id, pred, ins_label, out_loss, out_var, out_dist, out_reg):
        inds = tf.equal(i, unique_id)
        cur_pred = tf.boolean_mask(prediction, inds)
        cur_label = tf.boolean_mask(correct_label, inds)
        cur_discr_loss, cur_l_var, cur_l_dist, cur_l_reg = discriminative_loss_single(cur_pred, cur_label, feature_dim,
                                                                                      delta_v, delta_d, param_var,
                                                                                      param_dist, param_reg)
        out_loss = out_loss.write(i, cur_discr_loss)
        out_var = out_var.write(i, cur_l_var)
        out_dist = out_dist.write(i, cur_l_dist)
        out_reg = out_reg.write(i, cur_l_reg)

        return i + 1, ns, unique_id, pred, ins_label, out_loss, out_var, out_dist, out_reg

    output_ta_loss = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    output_ta_var = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    output_ta_dist = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    output_ta_reg = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    loop = [0, num_sems, unique_id, prediction, correct_label, output_ta_loss, output_ta_var, output_ta_dist,
            output_ta_reg]
    _, _, _, _, _, out_loss_op, out_var_op, out_dist_op, out_reg_op = tf.while_loop(cond, body, loop)

    out_loss_op = out_loss_op.stack()
    out_var_op = out_var_op.stack()
    out_dist_op = out_dist_op.stack()
    out_reg_op = out_reg_op.stack()

    disc_loss = tf.reduce_sum(out_loss_op)
    l_var = tf.reduce_sum(out_var_op)
    l_dist = tf.reduce_sum(out_dist_op)
    l_reg = tf.reduce_sum(out_reg_op)

    return disc_loss, l_var, l_dist, l_reg


def discriminative_loss_multicate(sem_label, prediction, correct_label, feature_dim,
                                  delta_v, delta_d, param_var, param_dist, param_reg):
    ''' Iterate over a batch of prediction/label and cumulate loss for multiple categories.
    :return: discriminative loss and its three components
    '''

    def cond(sem, label, batch, out_loss, out_var, out_dist, out_reg, i):
        return tf.less(i, tf.shape(batch)[0])

    def body(sem, label, batch, out_loss, out_var, out_dist, out_reg, i):
        disc_loss, l_var, l_dist, l_reg = discriminative_loss_single_multicate(sem_label[i], prediction[i],
                                                                               correct_label[i], feature_dim,
                                                                               delta_v, delta_d, param_var, param_dist,
                                                                               param_reg)

        out_loss = out_loss.write(i, disc_loss)
        out_var = out_var.write(i, l_var)
        out_dist = out_dist.write(i, l_dist)
        out_reg = out_reg.write(i, l_reg)

        return sem, label, batch, out_loss, out_var, out_dist, out_reg, i + 1

    # TensorArray is a data structure that support dynamic writing
    output_ta_loss = tf.TensorArray(dtype=tf.float32,
                                    size=0,
                                    dynamic_size=True)
    output_ta_var = tf.TensorArray(dtype=tf.float32,
                                   size=0,
                                   dynamic_size=True)
    output_ta_dist = tf.TensorArray(dtype=tf.float32,
                                    size=0,
                                    dynamic_size=True)
    output_ta_reg = tf.TensorArray(dtype=tf.float32,
                                   size=0,
                                   dynamic_size=True)

    _, _, _, out_loss_op, out_var_op, out_dist_op, out_reg_op, _ = tf.while_loop(cond, body, [sem_label,
                                                                                              correct_label,
                                                                                              prediction,
                                                                                              output_ta_loss,
                                                                                              output_ta_var,
                                                                                              output_ta_dist,
                                                                                              output_ta_reg,
                                                                                              0])
    out_loss_op = out_loss_op.stack()
    out_var_op = out_var_op.stack()
    out_dist_op = out_dist_op.stack()
    out_reg_op = out_reg_op.stack()

    disc_loss = tf.reduce_mean(out_loss_op)
    l_var = tf.reduce_mean(out_var_op)
    l_dist = tf.reduce_mean(out_dist_op)
    l_reg = tf.reduce_mean(out_reg_op)

    return disc_loss, l_var, l_dist, l_reg


def instance_center(prediction, correct_label, feature_dim, delta_v):
    """
    compute instance center point which has the max IoU in each instance
    :param prediction: inference network feature N*E
    :param correct_label: instance label N
    :param feature_dim: equal to E
    :param delta_v: the hinge to determine instance
    :return: correspond center to each point N*E
    """

    reshaped_pred = tf.reshape(prediction, [-1, feature_dim])
    num_point = reshaped_pred.get_shape().as_list()[0]

    # get the distance matrix between each point
    dist_mat = tf.reduce_sum(tf.abs(tf.subtract(reshaped_pred, tf.expand_dims(reshaped_pred, 1))), axis=2)
    dist_mat = tf.less(dist_mat, delta_v)

    # get the instance_label matrix
    correct_label_exp = tf.expand_dims(correct_label, -1)  # shape(n,1)
    correct_label_transpose = tf.transpose(correct_label_exp, perm=[1, 0])
    gt_mat = tf.subtract(correct_label_exp, correct_label_transpose)
    gt_mat = tf.equal(gt_mat, tf.constant(0))

    # compute IoU
    epsilon = tf.constant(tf.ones(num_point).astype(tf.float32) * 1e-6)
    pts_iou = tf.div(tf.reduce_sum(tf.cast(tf.logical_and(dist_mat, gt_mat), tf.float32), axis=1),
                     tf.reduce_sum(tf.cast(tf.logical_or(dist_mat, gt_mat), tf.float32), axis=1) + epsilon)

    # get the max IoU point and gather
    pts_iou = tf.expand_dims(pts_iou, 0)
    pts_iou = tf.tile(pts_iou, [num_point, 1])
    gt_mat = tf.cast(gt_mat, tf.float32)
    pts_iou = tf.multiply(pts_iou, gt_mat)

    pts_idx = tf.argmax(pts_iou, axis=1)
    pts_idx_unique, _ = tf.unique(pts_idx)
    center_feature_unique = tf.gather(prediction, pts_idx_unique)
    center_feature = tf.gather(prediction, pts_idx)

    return center_feature_unique, center_feature


def seed_cls_loss(prediction, ins_label, pred_seed, num_point, feature_dim, delta_v):
    reshaped_pred = tf.reshape(prediction, [-1, num_point, feature_dim])

    # get the distance matrix between each point
    dist_mat = tf.expand_dims(reshaped_pred, axis=2) - tf.expand_dims(reshaped_pred, axis=1)
    dist_mat = tf.norm(dist_mat, ord=1, axis=3)
    dist_mat = tf.less(dist_mat, delta_v)

    # get the instance_label matrix
    gt_mat = tf.expand_dims(ins_label, axis=-1) - tf.expand_dims(ins_label, axis=1)
    gt_mat = tf.equal(gt_mat, tf.constant(0))

    # compute IoU
    pts_iou = tf.div(tf.reduce_sum(tf.cast(tf.logical_and(dist_mat, gt_mat), tf.float32), axis=2),
                     tf.reduce_sum(tf.cast(tf.logical_or(dist_mat, gt_mat), tf.float32), axis=2))

    pts_iou_2 = 2 * tf.cast(tf.greater_equal(pts_iou, 0.9), dtype=tf.int32)
    pts_iou_1 = 1 * tf.cast(tf.greater_equal(pts_iou, 0.5) & tf.less(pts_iou, 0.9), dtype=tf.int32)
    pts_iou_0 = 0 * tf.cast(tf.less(pts_iou, 0.5), dtype=tf.int32)

    pts_label = pts_iou_0 + pts_iou_1 + pts_iou_2
    pts_label = tf.cast(pts_label, dtype=tf.int32)
    pts_label = tf.expand_dims(pts_label, -1)

    seed_loss = tf.losses.sparse_softmax_cross_entropy(labels=pts_label, logits=pred_seed)

    return seed_loss


def cosine_loss(prediction, ins_label, alpha=0.5):
    """
    :param prediction: tensor of similar feature B * N * E
    :param ins_label: ins_label B * N
    :param alpha: the hyperparameter of cos_distance
    :return: cosine_loss, same_loss, neq_loss
    """

    # dis_mat
    batch_size = prediction.shape()[0]
    num_point = prediction.shape()[1]
    prediction_norm = tf.sqrt(tf.reduce_sum(tf.square(prediction), axis=2))
    prediction_norm = tf.expand_dims(prediction_norm, axis=1)
    deno = tf.matmul(tf.transpose(prediction_norm, perm=[0, 2, 1]), prediction_norm)
    nume = tf.matmul(prediction, tf.transpose(prediction, perm=[0, 2, 1]))
    dis_mat = tf.divide(nume, deno)
    dis_mat = 0.5 * (1 + dis_mat)

    # same_mat
    same_mat = tf.expand_dims(prediction, axis=-1) - tf.expand_dims(prediction, axis=1)
    same_mat = tf.equal(same_mat, 0)
    neq_mat = tf.not_equal(same_mat, 0)

    # weight_mat (B * N * N)
    y, idx, count = tf.unique_with_counts(ins_label)
    count_exp = tf.gather(count, idx)
    weight_mat = tf.divide(1.0, count_exp)
    weight_mat = tf.matmul(tf.expand_dims(weight_mat, axis=-1), tf.expand_dims(weight_mat, axis=1))

    # same_loss & neq_loss
    same_loss = 1 - dis_mat
    same_loss = tf.multiply(same_loss, tf.cast(same_mat, tf.float32))
    same_loss = tf.multiply(same_loss, weight_mat)
    neq_loss = dis_mat - alpha
    neq_loss = tf.multiply(neq_loss, tf.cast(neq_mat, tf.float32))
    neq_loss = tf.clip_by_value(neq_loss, 0., neq_loss)
    neq_loss = tf.multiply(neq_loss, weight_mat)

    # total_loss
    loss_mat = same_loss + neq_loss
    cos_loss = tf.reduce_mean(loss_mat)
    same_loss = tf.reduce_sum(same_loss) / tf.reduce_sum(tf.cast(same_mat, dtype=tf.float32))
    neq_loss = tf.reduce_sum(neq_loss) / tf.reduce_sum(tf.cast(neq_mat, dtype=tf.float32))

    return cos_loss, same_loss, neq_loss




