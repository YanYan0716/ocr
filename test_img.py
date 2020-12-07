import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model

from DataPreprocess.imgUtil import ResizeImg
from Module.DetectBackbone import DetectModel

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
    print(f'img shape: {img.shape}')

    # 网络预测过程
    weight_dir = './model_weights/efficientnetb0/efficientnetb0_notop.h5'
    detectmodel = DetectModel()
    detect_eval_model = detectmodel.detect_eval_model(weights_dir=weight_dir,
                                                      model_name='EfficientNetB0',
                                                      imgs=img)
    # print(detect_eval_model.summary())
    # with tf.GradientTape() as tape:
    #     output = detect_eval_model(img)
    #
    # fts, endpoints = output[1:6], [output[0], output[-2]]
    # for i in range(len(fts)):
    #     print(fts[i].shape)
    # for i in range(len(endpoints)):
    #     print(endpoints[i].shape)