import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path.append('D:\\algorithm\\ocr')
sys.path.append('E:\\algorithm\\ocr')
sys.path.append('/content/ocr')

import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np

from Module.DetectLoss import detect_loss
'''
与原文改动：使用反卷积替换unpool
'''


@tf.function
def unpool(inputs):
    return tf.image.resize(inputs, size=[tf.shape(inputs)[1] * 2, tf.shape(inputs)[2] * 2])


class ConvBlock(layers.Layer):
    def __init__(self, output_num, kernel_size=1, activation_fn='relu', bn_flag=True, name=None):
        super(ConvBlock, self).__init__()
        k_initializer = tf.keras.initializers.truncated_normal()
        b_initializer = tf.keras.initializers.zeros()
        self.conv = layers.Conv2D(output_num,
                                  kernel_size,
                                  activation=activation_fn,
                                  kernel_initializer=k_initializer,
                                  kernel_regularizer=regularizers.L2(l2=1e-5),
                                  bias_initializer=b_initializer,
                                  padding='same',
                                  trainable=True,
                                  name=name
                                  )
        self.bn = layers.BatchNormalization(epsilon=1e-5, scale=True)
        self.bn_flag = bn_flag

    def call(self, inputs):
        x = self.conv(inputs)
        if self.bn_flag:
            x = self.bn(x)
        return x


class ConvBlock_develop(layers.Layer):
    def __init__(self, output_num):
        super(ConvBlock_develop, self).__init__()
        self.conv_block_1 = ConvBlock(output_num=output_num, kernel_size=1)
        self.conv_block_2 = ConvBlock(output_num=output_num, kernel_size=1)
        self.conv_block_3 = ConvBlock(output_num=output_num, kernel_size=3)

    def call(self, inp):
        inputs1 = inp[0]
        inputs2 = inp[1]
        x = self.conv_block_1(inputs1)
        x = tf.concat([inputs2, x], axis=-1)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        return x


class ContectBlock(layers.Layer):
    def __init__(self, num_outputs_recong=[256, 128, 64, 32, 32]):
        super(ContectBlock, self).__init__()
        self.num_outputs_recong = num_outputs_recong
        self.TEXT_SCALE = 512
        self.g = [None, None, None, None]
        self.h = [None, None, None, None]
        self.g_recong = [None, None, None, None, None]
        self.h_recong = [None, None, None, None, None]
        self.conv_develop_1 = [
            ConvBlock_develop(self.num_outputs_recong[1]),
            ConvBlock_develop(self.num_outputs_recong[2]),
            ConvBlock_develop(self.num_outputs_recong[3]),
        ]

        self.conv_develop_2 = [
            ConvBlock_develop(self.num_outputs_recong[2]),
            ConvBlock_develop(self.num_outputs_recong[3]),
            ConvBlock_develop(self.num_outputs_recong[4]),
        ]
        self.conv_h = ConvBlock(output_num=num_outputs_recong[0], kernel_size=1, name='conv_h')
        self.conv_g = ConvBlock(output_num=num_outputs_recong[3], kernel_size=3, name='conv_g')
        self.conv_h_recong = ConvBlock(output_num=num_outputs_recong[1], kernel_size=1, name='conv_h_recong')
        self.conv_g_recong = ConvBlock(output_num=num_outputs_recong[4], kernel_size=3, name='conv_g_recong')
        self.conv_sing_1_1 = ConvBlock(1, 1, activation_fn='sigmoid', bn_flag=False, name='conv_sing_1_1')
        self.conv_sing_4 = ConvBlock(4, 1, activation_fn='sigmoid', bn_flag=False, name='conv_sing_4')
        self.conv_sing_1_2 = ConvBlock(1, 1, activation_fn='sigmoid', bn_flag=False, name='conv_sing_1_2')

        self.unpool_1 = [
            keras.layers.Conv2DTranspose(num_outputs_recong[1], kernel_size=3, strides=2, padding='same'),
            keras.layers.Conv2DTranspose(num_outputs_recong[2], kernel_size=3, strides=2, padding='same'),
            keras.layers.Conv2DTranspose(num_outputs_recong[3], kernel_size=3, strides=2, padding='same'),
        ]

        self.unpool_2 = [
            keras.layers.Conv2DTranspose(num_outputs_recong[1], kernel_size=3, strides=2, padding='same'),
            keras.layers.Conv2DTranspose(num_outputs_recong[2], kernel_size=3, strides=2, padding='same'),
            keras.layers.Conv2DTranspose(num_outputs_recong[3], kernel_size=3, strides=2, padding='same'),
        ]

    def call(self, inputs):
        for i in range(4):
            if i == 0:
                self.h[i] = self.conv_h(inputs[i])
            else:
                self.h[i] = self.conv_develop_1[i - 1]([inputs[i], self.g[i - 1]])
            if i <= 2:
                self.g[i] = self.unpool_1[i](self.h[i])
            else:
                self.g[i] = self.conv_g(self.h[i])

        for i in range(1, 5):
            if i == 1:
                self.h_recong[i] = self.conv_h_recong(inputs[i])
            else:
                self.h_recong[i] = self.conv_develop_2[i - 2]([inputs[i], self.g_recong[i - 1]])
            if i <= 3:
                self.g_recong[i] = self.unpool_2[i - 2](self.h_recong[i])
            else:
                self.g_recong[i] = self.conv_g_recong(self.h_recong[i])

        F_score = self.conv_sing_1_1(self.g[3])
        geo_map = self.conv_sing_4(self.g[3]) * self.TEXT_SCALE
        angle_map = (self.conv_sing_1_2(self.g[3]) - 0.5) * np.pi / 2
        F_geometry = tf.concat([geo_map, angle_map], axis=-1)

        return self.g_recong[4], F_score, F_geometry
        # return self.g[3], F_score, F_geometry


class Detect_model(keras.Model):
    def __init__(self, base_weights_dir='./'):
        super(Detect_model, self).__init__()
        self.base_weights_dir = base_weights_dir
        self.base_model = keras.applications.EfficientNetB0(include_top=False,
                                                            weights=self.base_weights_dir,
                                                            # weights='imagenet',
                                                            input_tensor=None,
                                                            input_shape=None,
                                                            pooling='avg',
                                                            classes=1000,
                                                            classifier_activation=None
                                                            )
        self.base_results = [
            self.base_model.get_layer('normalization').output,
            self.base_model.get_layer('block1a_project_bn').output,
            self.base_model.get_layer('block2b_add').output,
            self.base_model.get_layer('block3b_add').output,
            self.base_model.get_layer('block5c_add').output,
            self.base_model.get_layer('block7a_project_bn').output,
            self.base_model.get_layer('avg_pool').output
        ]
        self.fts, self.endpoints = self.base_results[1:6][::-1], [self.base_results[0], self.base_results[-2]]

        self.develop_model = ContectBlock()

    def call(self, input_img):
        x = self.base_model(input_img)
        base_results = [
            self.base_model.get_layer('normalization').output,
            self.base_model.get_layer('block1a_project_bn').output,
            self.base_model.get_layer('block2b_add').output,
            self.base_model.get_layer('block3b_add').output,
            self.base_model.get_layer('block5c_add').output,
            self.base_model.get_layer('block7a_project_bn').output,
            self.base_model.get_layer('avg_pool').output
        ]
        fts, endpoints = base_results[1:6][::-1], [base_results[0], base_results[-2]]
        g_recong, F_score, F_geometry = self.develop_model(fts, training=True)
        return g_recong, F_score, F_geometry

    def model(self):
        return keras.Model(inputs=[self.base_model.inputs], outputs=self.call(self.base_model.inputs))


def loss(a, b, c):
    a = tf.reduce_mean(a[0])
    b = tf.reduce_mean(b[0])
    c = tf.reduce_mean(c[0])
    return tf.reduce_mean([a, b, c])


if __name__ == '__main__':
    weight_dir = './model_weights/efficientnetb0/efficientnetb0_notop.h5'
    detectmodel = Detect_model(base_weights_dir=weight_dir).model()
    optim = keras.optimizers.SGD(learning_rate=0.0001)

    # np.random.seed(1)
    # img = np.random.random((1, 512, 512, 3))
    # img = tf.convert_to_tensor(img, dtype=tf.float32)
    # score_maps = np.random.random((1, 128, 128, 1))
    # score_maps = tf.convert_to_tensor(score_maps, dtype=tf.float32)
    # training_masks = np.random.random((1, 128, 128, 1))
    # training_masks = tf.convert_to_tensor(training_masks, dtype=tf.float32)
    # geo_maps = np.random.random((1, 128, 128, 5))
    # geo_maps = tf.convert_to_tensor(geo_maps, dtype=tf.float32)

    data = np.load('./test_data.npy', allow_pickle=True)
    img = data[0]
    score_maps = data[1]
    geo_maps = data[2]
    training_masks = data[3]
    transform_matrixes = data[4]
    boxes_masks = data[5]
    box_widths = data[6]
    text_labels_sparse_0 = data[7]
    text_labels_sparse_1 = data[8]
    text_labels_sparse_2 = data[9]

    for i in range(0, 1):
        with tf.GradientTape() as tape:
            a, b, c = detectmodel(img, training=True)
            print(a.shape)
            print(b.shape)
            DetectLoss = detect_loss(tf.cast(score_maps, tf.float32),
                                     tf.cast(b, tf.float32),
                                     tf.cast(geo_maps, tf.float32),
                                     tf.cast(c, tf.float32),
                                     tf.cast(training_masks, tf.float32))
            print(DetectLoss)
        grad = tape.gradient([DetectLoss], detectmodel.trainable_weights)
        optim.apply_gradients(zip(grad, detectmodel.trainable_weights))

# ----------------以下为原来的做法，好像也不是原文中的------------
# import cv2
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers, regularizers
# import numpy as np
#
#
# def unpool(inputs):
#     return tf.image.resize(inputs, size=[tf.shape(inputs)[1] * 2, tf.shape(inputs)[2] * 2])
#
#
# class ConvBlock(layers.Layer):
#     def __init__(self, output_num, kernel_size=1, activation_fn='relu', bn_flag=True):
#         super(ConvBlock, self).__init__()
#         self.conv = layers.Conv2D(output_num,
#                                   kernel_size,
#                                   activation=activation_fn,
#                                   kernel_regularizer=regularizers.l2(1e-5),
#                                   padding='same',
#                                   trainable=True,
#                                   )
#         self.bn = layers.BatchNormalization(epsilon=1e-5, scale=True)
#         self.bn_flag = bn_flag
#
#     def call(self, inputs):
#         x = self.conv(inputs)
#         if self.bn_flag:
#             x = self.bn(x)
#         return x
#
#
# class ConvBlock_develop(layers.Layer):
#     def __init__(self, output_num):
#         super(ConvBlock_develop, self).__init__()
#         self.conv_block_1 = ConvBlock(output_num=output_num, kernel_size=1)
#         self.conv_block_2 = ConvBlock(output_num=output_num, kernel_size=1)
#         self.conv_block_3 = ConvBlock(output_num=output_num, kernel_size=3)
#
#     def call(self, inputs1, inputs2):
#         x = self.conv_block_1(inputs1)
#         x = tf.concat([inputs2, x], axis=-1)
#         x = self.conv_block_2(x)
#         x = self.conv_block_3(x)
#         return x
#
#
# class ContectBlock(layers.Layer):
#     def __init__(self, num_outputs_recong=[256, 128, 64, 32, 32]):
#         super(ContectBlock, self).__init__()
#         self.num_outputs_recong = num_outputs_recong
#         self.TEXT_SCALE = 512
#         self.g = [None, None, None, None]
#         self.h = [None, None, None, None]
#         self.g_recong = [None, None, None, None, None]
#         self.h_recong = [None, None, None, None, None]
#         self.conv_develop_1 = [
#             ConvBlock_develop(self.num_outputs_recong[0]),
#             ConvBlock_develop(self.num_outputs_recong[1]),
#             ConvBlock_develop(self.num_outputs_recong[2]),
#             ConvBlock_develop(self.num_outputs_recong[3]),
#         ]
#         self.conv_develop_2 = [
#             ConvBlock_develop(self.num_outputs_recong[0]),
#             ConvBlock_develop(self.num_outputs_recong[1]),
#             ConvBlock_develop(self.num_outputs_recong[2]),
#             ConvBlock_develop(self.num_outputs_recong[3]),
#             ConvBlock_develop(self.num_outputs_recong[4]),
#         ]
#         self.conv_h = ConvBlock(output_num=num_outputs_recong[0], kernel_size=1)
#         self.conv_g = ConvBlock(output_num=num_outputs_recong[3], kernel_size=3)
#         self.conv_h_recong = ConvBlock(output_num=num_outputs_recong[1], kernel_size=1)
#         self.conv_g_recong = ConvBlock(output_num=num_outputs_recong[4], kernel_size=3)
#         self.conv_sing_1_1 = ConvBlock(1, 1, activation_fn='sigmoid', bn_flag=False)
#         self.conv_sing_4 = ConvBlock(4, 1, activation_fn='sigmoid', bn_flag=False)
#         self.conv_sing_1_2 = ConvBlock(1, 1, activation_fn='sigmoid', bn_flag=False)
#
#     def call(self, inputs):
#         for i in range(4):
#             if i == 0:
#                 self.h[i] = self.conv_h(inputs[i])
#             else:
#                 self.h[i] = self.conv_develop_1[i](inputs[i], self.g[i-1])
#             if i <= 2:
#                 self.g[i] = unpool(self.h[i])
#             else:
#                 self.g[i] = self.conv_g(self.h[i])
#
#         for i in range(1, 5):
#             if i == 1:
#                 self.h_recong[i] = self.conv_h_recong(inputs[i])
#             else:
#                 self.h_recong[i] = self.conv_develop_2[i](inputs[i], self.g_recong[i-1])
#             if i <= 3:
#                 self.g_recong[i] = unpool(self.h_recong[i])
#             else:
#                 self.g_recong[i] = self.conv_g_recong(self.h_recong[i])
#
#         F_score = self.conv_sing_1_1(self.g[3])
#         geo_map = self.conv_sing_4(self.g[3]) * self.TEXT_SCALE
#         angle_map = (self.conv_sing_1_2(self.g[3]) - 0.5) * np.pi / 2
#         F_geometry = tf.concat([geo_map, angle_map], axis=-1)
#
#         return self.g_recong[4], F_score, F_geometry
#
#
# class Detect_model(keras.Model):
#     def __init__(self, base_weights_dir='./'):
#         super(Detect_model, self).__init__()
#         self.base_weights_dir = base_weights_dir
#         self.base_model = keras.applications.EfficientNetB0(include_top=False,
#                                                             weights=self.base_weights_dir,
#                                                             input_tensor=None,
#                                                             input_shape=None,
#                                                             pooling='avg',
#                                                             classes=1000,
#                                                             classifier_activation=None
#                                                             )
#         self.base_results = [
#             self.base_model.get_layer('normalization').output,
#             self.base_model.get_layer('block1a_project_bn').output,
#             self.base_model.get_layer('block2b_add').output,
#             self.base_model.get_layer('block3b_add').output,
#             self.base_model.get_layer('block5c_add').output,
#             self.base_model.get_layer('block7a_project_bn').output,
#             self.base_model.get_layer('avg_pool').output
#         ]
#         self.fts, self.endpoints = self.base_results[1:6][::-1], [self.base_results[0], self.base_results[-2]]
#
#         self.develop_model = ContectBlock()
#
#     def call(self, input_img):
#         x = self.base_model(input_img)
#         g_recong, F_score, F_geometry = self.develop_model(self.fts)
#         return g_recong, F_score, F_geometry
#
#     def model(self):
#         return keras.Model(inputs=[self.base_model.inputs], outputs=self.develop_model(self.fts))
#
#
# if __name__ == '__main__':
#     weight_dir = './model_weights/efficientnetb0/efficientnetb0_notop.h5'
#     detectmodel = Detect_model(base_weights_dir=weight_dir).model()
#
#     # 获取一张图片
#     img = 'img.jpg'
#
#     # 限制图片的长宽尺寸，保持原比例
#     img = cv2.imread(img)
#
#     # 交换颜色通道
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = tf.convert_to_tensor(img, dtype=tf.float32) / 255.0
#     img = tf.expand_dims(img, axis=0)
#     print(f'img shape: {img.shape}')
#
#     for layer in detectmodel.layers:
#         print(layer.name, layer.trainable)
#     print('***********************')
#     # detectmodel.summary()
#
#     a = detectmodel(img)[0]
#     print(a.shape)
