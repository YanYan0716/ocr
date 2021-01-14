import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path.append('D:\\algorithm\\ocr')
sys.path.append('E:\\algorithm\\ocr')
sys.path.append('/content/ocr/')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np


import config
from Module.RRotateLayer import RotateMyLayer
from Module.RecognitionLoss import recognition_loss


class CNNBlock(layers.Layer):
    def __init__(self, out_channels, cnn_kernel_size=3, pooling_kernel_size=[2, 1]):
        super(CNNBlock, self).__init__()
        self.conv1 = layers.Conv2D(filters=out_channels,
                                   kernel_size=cnn_kernel_size,
                                   kernel_regularizer=regularizers.l2(1e-5),
                                   data_format='channels_last',
                                   activation='relu',
                                   padding='same')
        self.bn = layers.BatchNormalization(epsilon=1e-5, scale=True, trainable=True, )
        self.conv2 = layers.Conv2D(filters=out_channels,
                                   kernel_size=cnn_kernel_size,
                                   kernel_regularizer=regularizers.l2(1e-5),
                                   data_format='channels_last',
                                   activation='relu',
                                   padding='same')
        self.pooling = layers.MaxPool2D(pool_size=pooling_kernel_size, strides=[2, 1], padding='same')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn(x)
        x = self.conv2(x)
        x = self.pooling(x)
        return x


class CNNEncoder(layers.Layer):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.group_num = 4
        self.cnn_1 = CNNBlock(64)
        self.cnn_2 = CNNBlock(128)
        self.cnn_3 = CNNBlock(256)
        self.cnn_list = [layers.Conv2D(filters=256 // self.group_num, kernel_size=[1, 1], padding='same'),
                         layers.Conv2D(filters=256 // self.group_num, kernel_size=[1, 3], padding='same'),
                         layers.Conv2D(filters=256 // self.group_num, kernel_size=[1, 5], padding='same'),
                         layers.Conv2D(filters=256 // self.group_num, kernel_size=[1, 7], padding='same'), ]
        self.cnn_out = layers.Conv2D(256, kernel_size=[1, 1], padding='same')

    def call(self, input_roi):
        y = self.cnn_1(input_roi)
        y = self.cnn_2(y)
        y = self.cnn_3(y)
        cnn_feat_list = []
        for i in range(self.group_num):
            cnn_feat_list.append(self.cnn_list[i](y))
        conv_final = tf.concat(cnn_feat_list, axis=-1)
        cnn_out = self.cnn_out(conv_final)
        return cnn_out


class LstmDecoder(layers.Layer):
    def __init__(self, lstm_hidden_num):
        super(LstmDecoder, self).__init__()
        self.lstm_hidden_num = lstm_hidden_num
        self.bilstm = layers.Bidirectional(
            layers.GRU(units=self.lstm_hidden_num,
                       dropout=0.8,
                       return_sequences=True,
                       return_state=False)
        )

    def call(self, input_tensor):
        batch_size = tf.shape(input_tensor)[0]
        lstm_output = self.bilstm(input_tensor)
        infer_output = tf.reshape(lstm_output, [-1, self.lstm_hidden_num * 2])

        W = tf.Variable(
            initial_value=lambda: tf.random.truncated_normal(
                shape=[self.lstm_hidden_num * 2, config.NUM_CLASSES],
                stddev=0.1),
            trainable=True)
        b = tf.Variable(initial_value=lambda: tf.constant(0., shape=[config.NUM_CLASSES]),
                        trainable=True)

        logits = tf.add(tf.matmul(infer_output, W), b)
        logits = tf.reshape(logits, [batch_size, -1, config.NUM_CLASSES])
        logits = tf.transpose(logits, (1, 0, 2))
        return logits
        # return output


class Recognition_model(keras.Model):
    def __init__(self, lstm_hidden_num):
        super(Recognition_model, self).__init__()
        self.lstm_hidden_num = lstm_hidden_num
        self.layer = RotateMyLayer()
        self.encoder = CNNEncoder()
        self.decoder = LstmDecoder(lstm_hidden_num=self.lstm_hidden_num)

    def call(self, x):
        a = self.layer(x)
        a = self.encoder(a)
        a = tf.squeeze(a, axis=1)
        # print(a.shape)
        a = self.decoder(a)
        return a

    def model(self):
        input1 = keras.Input(shape=(None, None, 32), dtype=tf.float32)
        input2 = keras.Input(shape=(None, 6), dtype=tf.float32)
        input3 = keras.Input(shape=(2,), dtype=tf.int32)
        return keras.Model(inputs=[input1, input2, input3],
                           outputs=self.call([input1, input2, input3]))


if __name__ == '__main__':
    shared_features = tf.random.normal([1, 256, 256, 32])
    input_transform_matrix = tf.random.normal([2, 6])

    input_box_masks = np.array([0., 0.])
    input_box_masks = tf.convert_to_tensor(input_box_masks, dtype=tf.float32)

    input_box_widths = np.array([17., 18.])
    input_box_widths = tf.convert_to_tensor(input_box_widths, tf.float32)

    reg_model = Recognition_model(lstm_hidden_num=256).model()

    text_labels_sparse_0 = np.array([
        [0,0], [0,1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2]
    ])
    text_labels_sparse_0 = tf.convert_to_tensor(text_labels_sparse_0, tf.int32)

    text_labels_sparse_1 = np.array([12, 12, 2, 2, 11, 30, 10])
    text_labels_sparse_1 = tf.convert_to_tensor(text_labels_sparse_1, tf.int32)

    text_labels_sparse_2 = np.array([2, 4])
    text_labels_sparse_2 = tf.convert_to_tensor(text_labels_sparse_2, tf.int32)

    optim = keras.optimizers.Adam(learning_rate=0.0001)

    for i in range(1):
        with tf.GradientTape() as tape:
            input_box_masks = tf.expand_dims(input_box_masks, axis=0)
            input_box_widths = tf.expand_dims(input_box_widths, axis=0)
            input_box_info = tf.transpose(tf.concat([input_box_masks, input_box_widths], axis=0))

            x = reg_model([shared_features, input_transform_matrix, input_box_info])

            RecognitionLoss = recognition_loss(x,
                                               text_labels_sparse_0,
                                               text_labels_sparse_1,
                                               text_labels_sparse_2,)
            RecognitionLoss = 0.01 * tf.cast(RecognitionLoss, dtype=tf.float32)
            print(RecognitionLoss)

        grad = tape.gradient([RecognitionLoss], reg_model.trainable_weights)

        grad = [tf.clip_by_norm(g, 2) for g in grad]
        optim.apply_gradients(zip(grad, reg_model.trainable_weights))




    # # text_labels_sparse.append([1,2,3,4,5,6])
    # text_labels_sparse.append([1, 1, 1, 1, 1, 1])
    # text_labels_sparse.append([3, 3])
    # labels = tf.sparse.SparseTensor(text_labels_sparse[0], text_labels_sparse[1], text_labels_sparse[2])
    # # box_widths=tf.cast(tf.convert_to_tensor([23, 12, 14]), tf.int32)
    # dd = tf.cast(tf.convert_to_tensor([192, 192, 192]), tf.int32)
    # loss = tf.nn.ctc_loss(labels, logits, label_length=None,logit_length= dd, blank_index=-1)
    # print(loss)
