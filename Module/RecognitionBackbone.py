import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys

sys.path.append('D:\\algorithm\\ocr')
sys.path.append('E:\\algorithm\\ocr')
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np


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
        # print(f'inputs shape :{inputs.shape}')
        x = self.conv1(inputs)
        # print(f'x shape :{x.shape}')
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
        self.bilstm = layers.Bidirectional(layers.GRU(units=self.lstm_hidden_num,
                                                       dropout=0.8,
                                                       return_state=True,
                                                       return_sequences=True))

    def call(self, input_tensor):
        output = self.bilstm(input_tensor)
        # infer_output = tf.concat(output, axis=-1)

        # return infer_output
        return output


class Recognition_model(keras.Model):
    def __init__(self, lstm_hidden_num, training=False):
        super(Recognition_model, self).__init__()
        self.lstm_hidden_num = lstm_hidden_num
        self.training = training
        self.encoder = CNNEncoder()
        self.decoder = LstmDecoder(lstm_hidden_num=self.lstm_hidden_num)

    def call(self, roi_fmp):
        a = self.encoder(roi_fmp)
        a = tf.squeeze(a, axis=1)
        # a = self.decoder(a)
        return a

    def model(self):
        x = keras.Input(shape=[None, None, 32])
        return keras.Model(inputs=[x], outputs=self.call(x))


if __name__ == '__main__':
    roi_tensor = tf.random.normal([3, 8, 384, 32])
    reg_model = Recognition_model(lstm_hidden_num=256).model()
    x = reg_model.predict_step(roi_tensor)
    print(x.shape)


    '''
    logits = tf.zeros([192, 3, 94])
    text_labels_sparse=[]
    a=np.array([0, 0])
    text_labels_sparse.append(np.array([[0, 0],
                                      [0, 1],
                                      [0, 2],
                                      [1, 0],
                                      [1, 1],
                                      [2, 0]]))
    # text_labels_sparse.append([1,2,3,4,5,6])
    text_labels_sparse.append([1, 1, 1, 1, 1, 1])
    text_labels_sparse.append([3, 3])
    labels = tf.sparse.SparseTensor(text_labels_sparse[0], text_labels_sparse[1], text_labels_sparse[2])
    # box_widths=tf.cast(tf.convert_to_tensor([23, 12, 14]), tf.int32)
    dd = tf.cast(tf.convert_to_tensor([192, 192, 192]), tf.int32)
    loss = tf.nn.ctc_loss(labels, logits, label_length=None,logit_length= dd, blank_index=-1)
    print(loss)
    '''



    import tensorflow as tf

    label1 = 'ab'
    label2 = 'c'
    label_matrix = np.array([[0, 0]])
    value = tf.convert_to_tensor(np.array([0]))
    shape = tf.cast(tf.convert_to_tensor(np.array([1, 1])), dtype=tf.int64)
    label_matrix = tf.SparseTensor(tf.cast(tf.convert_to_tensor(label_matrix), dtype=tf.int64),
                                   values=value,
                                   dense_shape=shape,)

    logits = np.array([[[0., 0., 1.], [0., 1., 0.]]])
    a=tf.nn.softmax(logits)
    print(a)

    # label = [0]
    # logits = (np.array([[[0.7, 0.3]]]))
    # labels_length = 1
    # print(logits.shape)
    # labels_tensor = tf.convert_to_tensor([label], dtype=tf.int32)
    logits_tensor = tf.convert_to_tensor(logits, dtype=tf.float32)

    print(logits_tensor.shape)
    # labels_length_tensor = tf.convert_to_tensor([labels_length], dtype=tf.int32)
    logit_length = tf.fill([tf.shape(logits_tensor)[0]], tf.shape(logits_tensor)[1])
    print(logit_length)
    loss = tf.nn.ctc_loss(label_matrix, logits_tensor, label_length=None, logit_length= logit_length,
                          logits_time_major=False, blank_index=tf.convert_to_tensor(2, dtype=tf.int32))
    # loss = tf.reduce_mean(loss)
    print(loss.numpy())