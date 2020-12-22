import tensorflow as tf


def recognition_loss(logits, labels, logits_length, label_length):
    loss = tf.nn.ctc_loss(labels, logits, label_length, logits_length)
    loss = tf.reduce_mean(loss)
    return loss


def decode(logits, seq_len):
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len, merge_repeated=True)
    # 有问题。。。。。。。。。
    a = tf.sparse.SparseTensor(decoded[0])
    dense_decoded = tf.sparse.to_dense(a, default_value=-1) # 有问题，？？？？？？？？？？？
    return decoded, dense_decoded