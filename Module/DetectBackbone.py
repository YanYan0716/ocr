import tensorflow as tf
from Module.efficientnet.lite.efficientnet_lite_builder import build_model_base


def normalize(image):
    return tf.cast(image, tf.float32) / 255.0


class Backbone(object):
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
