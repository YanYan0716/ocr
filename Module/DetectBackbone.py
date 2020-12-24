import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np


def unpool(inputs):
    return tf.image.resize(inputs, size=[tf.shape(inputs)[1] * 2, tf.shape(inputs)[2] * 2])


class ConvBlock(layers.Layer):
    def __init__(self, output_num, kernel_size=1, activation_fn='relu', bn_flag=True):
        super(ConvBlock, self).__init__()
        self.conv = layers.Conv2D(output_num,
                                  kernel_size,
                                  activation=activation_fn,
                                  kernel_regularizer=regularizers.l2(1e-5),
                                  padding='same'
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

    def call(self, inputs1, inputs2):
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
            ConvBlock_develop(self.num_outputs_recong[0]),
            ConvBlock_develop(self.num_outputs_recong[1]),
            ConvBlock_develop(self.num_outputs_recong[2]),
            ConvBlock_develop(self.num_outputs_recong[3]),
        ]
        self.conv_develop_2 = [
            ConvBlock_develop(self.num_outputs_recong[0]),
            ConvBlock_develop(self.num_outputs_recong[1]),
            ConvBlock_develop(self.num_outputs_recong[2]),
            ConvBlock_develop(self.num_outputs_recong[3]),
            ConvBlock_develop(self.num_outputs_recong[4]),
        ]
        self.conv_h = ConvBlock(output_num=num_outputs_recong[0], kernel_size=1)
        self.conv_g = ConvBlock(output_num=num_outputs_recong[3], kernel_size=3)
        self.conv_h_recong = ConvBlock(output_num=num_outputs_recong[1], kernel_size=1)
        self.conv_g_recong = ConvBlock(output_num=num_outputs_recong[4], kernel_size=3)
        self.conv_sing_1_1 = ConvBlock(1, 1, activation_fn='sigmoid', bn_flag=False)
        self.conv_sing_4 = ConvBlock(4, 1, activation_fn='sigmoid', bn_flag=False)
        self.conv_sing_1_2 = ConvBlock(1, 1, activation_fn='sigmoid', bn_flag=False)

    def call(self, inputs):
        for i in range(4):
            if i == 0:
                self.h[i] = self.conv_h(inputs[i])
            else:
                self.h[i] = self.conv_develop_1[i](inputs[i], self.g[i-1])
            if i <= 2:
                self.g[i] = unpool(self.h[i])
            else:
                self.g[i] = self.conv_g(self.h[i])

        for i in range(1, 5):
            if i == 1:
                self.h_recong[i] = self.conv_h_recong(inputs[i])
            else:
                self.h_recong[i] = self.conv_develop_2[i](inputs[i], self.g_recong[i-1])
            if i <= 3:
                self.g_recong[i] = unpool(self.h_recong[i])
            else:
                self.g_recong[i] = self.conv_g_recong(self.h_recong[i])

        F_score = self.conv_sing_1_1(self.g[3])
        geo_map = self.conv_sing_4(self.g[3]) * self.TEXT_SCALE
        angle_map = (self.conv_sing_1_2(self.g[3]) - 0.5) * np.pi / 2
        F_geometry = tf.concat([geo_map, angle_map], axis=-1)

        return self.g_recong[4], F_score, F_geometry


class Detect_model(keras.Model):
    def __init__(self, base_weights_dir='./'):
        super(Detect_model, self).__init__()
        self.base_weights_dir = base_weights_dir
        self.base_model = keras.applications.EfficientNetB0(include_top=False,
                                                            weights=self.base_weights_dir,
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
        g_recong, F_score, F_geometry = self.develop_model.call(self.fts)
        return g_recong, F_score, F_geometry

    def model(self):
        # base_results = [
        #     self.base_model.get_layer('normalization').output,
        #     self.base_model.get_layer('block1a_project_bn').output,
        #     self.base_model.get_layer('block2b_add').output,
        #     self.base_model.get_layer('block3b_add').output,
        #     self.base_model.get_layer('block5c_add').output,
        #     self.base_model.get_layer('block7a_project_bn').output,
        #     self.base_model.get_layer('avg_pool').output]
        # fts, endpoints = base_results[1:6][::-1], [base_results[0], base_results[-2]]
        return keras.Model(inputs=self.base_model.inputs, outputs=self.develop_model(self.fts))


if __name__ == '__main__':
    weight_dir = './model_weights/efficientnetb0/efficientnetb0_notop.h5'
    detectmodel = Detect_model(base_weights_dir=weight_dir).model()

    for layer in detectmodel.layers:
        layer.trainable = False
    detectmodel.layers[-1].trainable = True
    # 获取一张图片
    img = 'img.jpg'

    # 限制图片的长宽尺寸，保持原比例
    img = cv2.imread(img)

    # 交换颜色通道
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tf.convert_to_tensor(img, dtype=tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)
    print(f'img shape: {img.shape}')

    # for layer in detectmodel.layers:
    #     print(layer.name, layer.trainable)
    # print('***********************')
    # detectmodel.summary()

    a = detectmodel(img)[0]
    # for i in range(len(a)):
    #     print(a[i].shape)
    print(a.shape)

# **********************************以下为原始方法，可作参考，但是已经

#
# def normalize(image):
#     return tf.cast(image, tf.float32) / 255.0
#
#
# def my_conv(inputs, output_num, kernel_size=1, activation_fn='relu', bn=True):
#     x = layers.Conv2D(output_num,
#                       kernel_size,
#                       activation=activation_fn,
#                       kernel_regularizer=regularizers.l2(1e-5),
#                       trainable=False,
#                       padding='same')(inputs)
#     if bn == True:
#         x = layers.BatchNormalization(epsilon=1e-5, scale=True, trainable=False)(x)
#     return x
#
#
# class DetectModel(object):
#     def __init__(self, is_training=True):
#         self.is_training = is_training
#
#     def model(self, model_name, weights_dir):
#         '''
#         定义检测的模型
#         :param imgs:
#         :param weight_decay:
#         :return:
#         '''
#         # 图片归一化
#         # img = normalize(imgs)
#
#         return model_name, weights_dir
#
#     def detect_eval_model(self, weights_dir, model_name, imgs=None):
#         assert model_name == 'EfficientNetB0', 'detect model is wrong'
#         base_model = keras.applications.EfficientNetB0(
#             include_top=False,
#             weights=weights_dir,
#             input_tensor=None,
#             input_shape=None,
#             pooling='avg',
#             classes=1000,
#             classifier_activation=None
#         )
#         model = Model(
#             inputs=base_model.input,
#             outputs=[
#                 base_model.get_layer('normalization').output,
#                 base_model.get_layer('block1a_project_bn').output,
#                 base_model.get_layer('block2b_add').output,
#                 base_model.get_layer('block3b_add').output,
#                 base_model.get_layer('block5c_add').output,
#                 base_model.get_layer('block7a_project_bn').output,
#                 base_model.get_layer('avg_pool').output,
#             ]
#         )
#
#         with tf.GradientTape() as tape:
#             # 图片归一化
#             imgs = normalize(imgs)
#             output = model(imgs)
#         fts, endpoints = output[1:6][::-1], [output[0], output[-2]]
#
#         g = [None, None, None, None]
#         h = [None, None, None, None]
#
#         g_recong = [None, None, None, None, None]
#         h_recong = [None, None, None, None, None]
#         num_outputs_recong = [256, 128, 64, 32, 32]
#
#         for i in range(4):
#             if i == 0:
#                 h[i] = my_conv(fts[i], num_outputs_recong[i], 1)
#             else:
#                 f_t = my_conv(fts[i], num_outputs_recong[i], 1)
#                 print(f'f_t shape {f_t.shape}')
#                 print(num_outputs_recong[i])
#                 c1_1 = my_conv(tf.concat([g[i - 1], f_t], axis=-1),
#                                num_outputs_recong[i],
#                                1)
#                 h[i] = my_conv(c1_1, num_outputs_recong[i], 3)
#
#             if i <= 2:
#                 g[i] = unpool(h[i])
#                 print(f'detect {g[i].shape}')
#             else:
#                 g[i] = my_conv(h[i], num_outputs_recong[i], 3)
#
#         for i in range(1, 5):
#             if i == 1:
#                 h_recong[i] = my_conv(fts[i], num_outputs_recong[i], 1)
#             else:
#                 f_t = my_conv(fts[i], num_outputs_recong[i], 1)
#                 c1_1_recong = my_conv(tf.concat([g_recong[i - 1], f_t], axis=-1),
#                                       num_outputs_recong[i],
#                                       1)
#                 h_recong[i] = my_conv(c1_1_recong, num_outputs_recong[i], 3)
#
#             if i <= 3:
#                 g_recong[i] = unpool(h_recong[i])
#             else:
#                 g_recong[i] = my_conv(h_recong[i], num_outputs_recong[i], 3)
#
#         F_score = my_conv(g[3], 1, 1, activation_fn='sigmoid', bn=False)
#         TEXT_SCALE = 512
#         geo_map = my_conv(g[3], 4, 1, activation_fn='sigmoid', bn=False) * TEXT_SCALE
#         angle_map = (my_conv(g[3], 1, 1, activation_fn='sigmoid', bn=False) - 0.5) * np.pi / 2
#         F_geometry = tf.concat([geo_map, angle_map], axis=-1)
#
#         return g_recong[4], F_score, F_geometry