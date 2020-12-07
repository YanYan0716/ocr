import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import Model
from tensorflow.keras import layers, regularizers
import numpy as np


from Module.efficientnet.lite.efficientnet_lite_builder import build_model_base


def unpool(inputs):
    return tf.image.resize(inputs, size=[tf.shape(inputs)[1]*2, tf.shape(inputs)[2]*2])


def normalize(image):
    return tf.cast(image, tf.float32) / 255.0


def my_conv(inputs, output_num, keernel_size=1, activation_fc='relu', bn=True):
    x = layers.Conv2D(output_num, keernel_size, activation=activation_fc, kernel_regularizer=regularizers.l2(1e-5))(inputs)
    if bn==True:
        x = layers.BatchNormalization(epsilon=1e-5, scale=True)(x)
    return x


class DetectModel(object):
    def __init__(self, is_training=True):
        self.is_training = is_training

    def model(self, imgs, weight_decay=1e-5):
        '''
        定义检测的模型
        :param imgs:
        :param weight_decay:
        :return:
        '''
        # 图片归一化
        img = normalize(imgs)

        fts, endpoints = build_model_base(img, 'efficientnet-lite0', self.is_training)
        return fts, endpoints

    def detect_eval_model(self, weights_dir, model_name, imgs=None):
        assert model_name == 'EfficientNetB0', 'detect model is wrong'
        base_model = keras.applications.EfficientNetB0(
            include_top=False,
            weights=weights_dir,
            input_tensor=None,
            input_shape=None,
            pooling='avg',
            classes=1000,
            classifier_activation=None
        )
        model = Model(
            inputs=base_model.input,
            outputs=[
                base_model.get_layer('normalization').output,
                base_model.get_layer('block1a_project_bn').output,
                base_model.get_layer('block2b_add').output,
                base_model.get_layer('block3b_add').output,
                base_model.get_layer('block5c_add').output,
                base_model.get_layer('block7a_project_bn').output,
                base_model.get_layer('avg_pool').output,
            ]
        )

        with tf.GradientTape() as tape:
            # 图片归一化
            imgs = normalize(imgs)
            output = model(imgs)
        fts, endpoints = output[1:6], [output[0], output[-2]]

        g = [None, None, None, None]
        h = [None, None, None, None]

        g_recong = [None, None, None, None, None]
        h_recong = [None, None, None, None, None]
        num_outputs_recong = [256, 128, 64, 32, 32]

        for i in range(4):
            if i == 0:
                h[i] = my_conv(fts[i], num_outputs_recong[i], 1)
            else:
                f_t = my_conv(fts[i], num_outputs_recong[i], 1)
                c1_1 = my_conv(tf.concat([g[i-1], f_t], axis=-1), num_outputs_recong[i], 1)
                h[i] = my_conv(c1_1, num_outputs_recong[i], 3)

            if i<=2:
                g_recong[i] = unpool(h_recong[i])
            else:
                g_recong[i] = my_conv(h_recong[i], num_outputs_recong[i], 3)

        F_score = my_conv(g[3],1,1, activation_fc='sigmoid', bn=False)
        TEXT_SCALE = 512
        geo_map = my_conv(g[3], 4, 1, activation_fc='sigmoid', bn=False)*TEXT_SCALE
        angle_map = (my_conv(g[3], 1, 1, activation_fn='sigmoid', bn=False) - 0.5) * np.pi / 2
        F_geometry = tf.concat([geo_map, angle_map], axis=-1)
        return g_recong[4], F_score, F_geometry