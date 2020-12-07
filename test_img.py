import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import tensorflow as tf

from DataPreprocess.imgUtil import ResizeImg
from Module.DetectBackbone import Backbone

if __name__ == '__main__':
    # 获取一张图片
    img = 'img.jpg'

    # 限制图片的长宽尺寸，保持原比例
    img = cv2.imread(img)
    img, (ratio_h, ratio_w) = ResizeImg(img)

    # 交换颜色通道
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tf.convert_to_tensor(img, dtype=tf.uint8)
    img = tf.expand_dims(img, axis=0)
    print(img.shape)

    # 网络预测过程
    model_path = './model_weights/efficientnet-lite0/model.ckpt'
    detect_part = Backbone(is_training=False)
    fts, endpoints = detect_part.model(img)
    print(fts.shape)
    print(endpoints['reduction_1'].shape)
    print(endpoints['reduction_2'].shape)
    print(endpoints['reduction_3'].shape)
    print(endpoints['reduction_4'].shape)
    print(endpoints['reduction_5'].shape)

    # step 1: detection
