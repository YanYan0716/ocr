import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

import sys
sys.path.append('D:\\algorithm\\ocr')
sys.path.append('E:\\algorithm\\ocr')
sys.path.append('/home/epai/yanqian/ocr')
import config


def recognition_loss(logits, labels_indices, labels_values, labels_dense_shape):
    labels_sparse_tensor = tf.SparseTensor(indices=tf.cast(labels_indices, tf.int64),
                                           values=labels_values,
                                           dense_shape=tf.cast(labels_dense_shape, tf.int64))
    log_len = []
    for index in range(logits.shape[1]):
        log_len.append(config.NUM_CLASSES)
    logits_length = tf.convert_to_tensor(log_len, dtype=tf.int32)

    loss = tf.nn.ctc_loss(labels=labels_sparse_tensor,
                          logits=logits,
                          label_length=None,
                          logit_length=logits_length,
                          blank_index= -1,
                          logits_time_major=True)
    loss = tf.reduce_mean(loss)
    return loss


def decode(logits, seq_len):
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len, merge_repeated=True)
    # print(len(decoded))
    print(decoded[0].shape)
    dense_decoded = tf.sparse.to_dense(decoded[0], default_value=-1) # 有问题，？？？？？？？？？？？
    return dense_decoded


if __name__ == '__main__':
    # 检测decode函数的正确性, 默认将最后tensor中每个time step的最后一位数值代表black index
    tensor = np.zeros((384, 95))
    tensor[:, -1] = tensor[:, -1]+1
    tensor[0][1] = 2
    tensor[1][10] = 4
    print(tensor[10])
    tensor = tf.convert_to_tensor(tensor, dtype=tf.float32)

    tensor = tf.expand_dims(tensor, axis=1)
    seq_len = [95]  # 代表了一共有多少个分类
    seq_len = tf.convert_to_tensor(seq_len, dtype=tf.int32)
    result = decode(tensor, seq_len)
    print(result)



