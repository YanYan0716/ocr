import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers


if __name__ == '__main__':
    forward_layer = layers.LSTMCell(256, dropout=0.8, recurrent_dropout=0.8)
    backward_layer = layers.LSTMCell(256, dropout=0.8, recurrent_dropout=0.8)

    # forward_layer = layers.LSTMCell(256)
    # backward_layer = layers.LSTMCell(256)
    #
    # forward_layer = tf.nn.RNNCellDropoutWrapper(
    #     forward_layer,
    #     input_keep_prob=0.8,
    #     output_keep_prob=0.8
    # )
    #
    # backward_layer = tf.nn.RNNCellDropoutWrapper(
    #     backward_layer,
    #     input_keep_prob=0.8,
    #     output_keep_prob=0.8
    # )

    forward_layer = layers.RNN(forward_layer, return_sequences=True)
    backward_layer = layers.RNN(backward_layer, return_sequences=True, go_backwards=True)

    model = layers.Bidirectional(forward_layer, backward_layer=backward_layer)

    shared_features = tf.random.normal([3, 384, 256])
    output = model(shared_features)
    # infer_output = tf.reshape(output, [-1, 256 * 2])
    print(output.shape)