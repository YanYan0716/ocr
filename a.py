import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers


def dice_coefficient(true_cls, pred_cls, training_mask):
    '''
    最理想的情况是true和pred相等，此时loss应该为0 所以在loss=...的那一行有一个scale：2
    :param true_cls:
    :param pred_cls:
    :param training_mask:
    :return:
    '''
    eps = 1e-5  # 保证除法中分母不为0
    inter = tf.reduce_sum(true_cls * pred_cls * training_mask)
    print(tf.reduce_sum(true_cls*pred_cls))
    print(inter)
    union_1 = tf.reduce_sum(true_cls * training_mask)
    print(union_1)
    union_2 = tf.reduce_sum(pred_cls * training_mask)
    print(union_2)
    union = (union_1 + union_2) + eps
    loss = 1. - (2 * inter / union)
    return loss


if __name__ == '__main__':
    test1 = np.load('./test_data_test_0.npy', allow_pickle=True)
    score_map = test1[1]
    training_mask = test1[3]

    print(tf.reduce_sum(score_map))
    print(tf.reduce_sum(training_mask))
    print(tf.reduce_sum(score_map*score_map))

    DetectLoss = dice_coefficient(tf.cast(score_map, tf.float32),
                             tf.cast(score_map, tf.float32),
                             tf.cast(training_mask, tf.float32))
    # print(DetectLoss)