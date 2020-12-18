import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

import config


class CNNBlock(layers.Layer):
    def __init__(self, out_channels, cnn_kernel_size=3, pooling_kernel_size=[2, 1]):
        super(CNNBlock, self).__init__()
        self.conv = layers.Conv2D(filters=out_channels,
                                  kernel_size=cnn_kernel_size,
                                  kernel_regularizer=regularizers.l2(1e-5),
                                  data_format='NHWC',
                                  activation='relu',
                                  padding='same')
        self.bn = layers.BatchNormalization(epsilon=1e-5, scale=True, trainable=True, )
        self.pooling = layers.MaxPool2D(pool_size=pooling_kernel_size, stride=[2, 1], padding='same')

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.conv(x)


class CNNEncoder(layers.Layer):
    def __init__(self):
        super(CNNEncoder, self).__init__
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
        x = self.cnn_1(input_roi)
        x = self.cnn_2(x)
        x = self.cnn_3(x)
        cnn_feat_list = []
        for i in range(self.group_num):
            cnn_feat_list.append(self.cnn_list[i](x))
        conv_final = tf.concat(cnn_feat_list, axis=-1)
        cnn_out = self.cnn_out(conv_final)
        return cnn_out


class LstmDecoder(layers.Layer):
    def __init__(self, lstm_hidden_num):
        super(LstmDecoder, self).__init__
        self.lstm_hidden_num = lstm_hidden_num
        self.bilstm = layers.Bidirectional(layers.LSTM(units=self.lstm_hidden_num,
                                                       dropout=0.8,
                                                       return_state=True,
                                                       return_sequences=True))

    def call(self, input_tensor):
        output, state = self.bilstm(input_tensor)
        infer_output = tf.concat(output, axis=-1)

        return infer_output


class Recognition_model(keras.Model):
    def __init__(self, lstm_hidden_num, training=False):
        self.lstm_hidden_num = lstm_hidden_num
        self.training = training
        self.encoder = CNNEncoder()
        self.decoder = LstmDecoder(lstm_hidden_num=self.lstm_hidden_num)

    def call(self, roi_fmp):
        x = self.encoder(roi_fmp)
        x = self.decoder(x)
        return x

    def model(self):
        x = keras.Input(shape=(224, 224, 3))
        return keras.Model(inputs=[x], outputs=self.call(x))


if __name__ == '__main__':
    roi_tensor = tf.random.normal([3, 8, 384, 32])
    reg_model = Recognition_model(lstm_hidden_num=256).model()
    x = reg_model.predict_step(roi_tensor)
