import tensorflow as tf


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
    # 有问题。。。。。。。。。
    a = tf.sparse.SparseTensor(decoded[0])
    dense_decoded = tf.sparse.to_dense(a, default_value=-1) # 有问题，？？？？？？？？？？？
    return decoded, dense_decoded
