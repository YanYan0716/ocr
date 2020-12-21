import tensorflow as tf


def recognition_loss(logits, labels, logits_length, label_length):
    loss = tf.nn.ctc_loss(labels, logits, label_length, logits_length)
    loss = tf.reduce_mean(loss)
    return loss


def decode(logits, seq_len):
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len, merge_repeated=True)
    dense_decoded = tf.sparse() # 有问题，不对
    return decoded, dense_decoded