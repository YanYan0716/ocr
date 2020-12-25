import sys

sys.path.append('D:\\algorithm\\ocr')
sys.path.append('E:\\algorithm\\ocr')
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as keras
import numpy as np

from Module.transformer import spatial_transformer_network as transformer


def func(x):
    b = np.min((30, 200))
    for i in range(x.shape[1]):
        x += 1
    return x+b


class RotateMyLayer(layers.Layer):
    def __init__(self):
        super(RotateMyLayer, self).__init__()

    def call(self, shared_features,  box_info):
        converted_f = tf.autograph.to_graph(func)
        a = converted_f(box_info)
        print(a)
        return box_info


class RotateModel(keras.Model):
    def __init__(self):
        super(RotateModel, self).__init__()
        self.layer = RotateMyLayer()

    def call(self, shared_features, input_box_info):
        a = self.layer(shared_features, input_box_info)
        return a

    def model(self):
        input1 = keras.Input(shape=(None, None, 32), dtype=tf.float32)
        # input2 = keras.Input(shape=[None, 6], dtype=tf.float32)
        input3 = keras.Input(shape=(2, ), dtype=tf.int32)
        return keras.Model(inputs=[input1, input3],
                           outputs=self.call(input1, input3))


if __name__ == '__main__':
    # (3, 8, 384, 32)

    shared_features = tf.random.normal([2, 112, 112, 32])
    input_transform_matrix = tf.random.normal([3, 6])
    input_box_masks = tf.expand_dims(tf.convert_to_tensor([2, 2, 1]), axis=0)
    input_box_widths = tf.expand_dims(tf.convert_to_tensor([55, 12, 13]), axis=0)
    input_box_info = tf.transpose(tf.concat([input_box_masks, input_box_widths], axis=0))

    model = RotateModel().model()

    output = model([shared_features, input_box_widths, input_box_info])
    # print(output.shape)
