import sys

sys.path.append('D:\\algorithm\\ocr')
sys.path.append('E:\\algorithm\\ocr')
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as keras
import numpy as np

from Module.transformer import spatial_transformer_network as transformer

@tf.function
def func(shared_features, transform_matrix, box_info):
    box_nums = transform_matrix.shape[0]
    box_masks = box_info[:, 0]
    box_widths = box_info[:, 1]

    if box_nums:
        pad_rois = tf.TensorArray(tf.float32, size=box_nums)
        max_width = 384
        i = 0
        def cond(pad_rois, i):
            return i < box_nums

        def body(pad_rois, i):
            index = box_masks[i]
            matrix = transform_matrix[i]
            _feature_map = shared_features[index]
            box_width = box_widths[i]

            ex_feature_maps = tf.expand_dims(_feature_map, 0)
            trans_matrix = tf.expand_dims(matrix, 0)

            trans_feature_map = transformer(ex_feature_maps, trans_matrix, [8, box_width])

            roi = tf.image.crop_and_resize(trans_feature_map,  # 将每个trans_feature_map resize到[8, box_width]
                                           [[0.0, 0.0, 1.0, 1.0]],
                                           [0],
                                           [8, box_width],
                                           method='nearest',
                                           name='crop')

            pad_roi = tf.image.pad_to_bounding_box(roi,  # 将每个roi resize到[8, 384]
                                                   0,
                                                   0,
                                                   8,
                                                   max_width)

            pad_rois = pad_rois.write(i, pad_roi)
            i += 1
            return pad_rois, i
        pad_rois, _ = tf.while_loop(cond, body, loop_vars=[pad_rois, i], parallel_iterations=10, swap_memory=True)
        pad_rois = pad_rois.stack()
        return tf.squeeze(pad_rois, axis=1)
    else:
        print('something in Roirotate')


class RotateMyLayer(layers.Layer):
    def __init__(self):
        super(RotateMyLayer, self).__init__(name='roitate', trainable=False)

    def call(self, x):
        shared_features = x[0]
        transform_matrix = x[1]
        box_info = x[2]
        result = func(shared_features, transform_matrix, box_info)
        return result


class RotateModel(keras.Model):
    def __init__(self):
        super(RotateModel, self).__init__()
        self.layer = RotateMyLayer()

    def call(self, x):
        a = self.layer(x)
        print(type(a))
        if a.values():
            print('ok')
        else:
            print('nono')
        # print(a.values())
        # a = layers.Conv2D(3, 3, 3)(a)
        return a

    def model(self):
        input1 = keras.Input(shape=(None, None, 32), dtype=tf.float32)
        input2 = keras.Input(shape=(None, 6), dtype=tf.float32)
        input3 = keras.Input(shape=(2, ), dtype=tf.int32)
        return keras.Model(inputs=[input1, input2, input3],
                           outputs=self.call([input1, input2, input3]))


def loss(a):
    a = tf.reduce_mean(a[0])
    return a


if __name__ == '__main__':
    # (3, 8, 384, 32)

    shared_features = tf.random.normal([2, 112, 112, 32])
    input_transform_matrix = tf.random.normal([3, 6])
    input_box_masks = tf.expand_dims(tf.convert_to_tensor([0, 0, 1]), axis=0)
    input_box_widths = tf.expand_dims(tf.convert_to_tensor([55, 12, 13]), axis=0)
    input_box_info = tf.transpose(tf.concat([input_box_masks, input_box_widths], axis=0))

    model = RotateModel().model()
    optim = keras.optimizers.Adam()

    with tf.GradientTape() as tape:
        output = model([shared_features, input_transform_matrix,  input_box_info])
        print(output.shape)
        loss_num = loss(output)
        print(loss_num)
    grad = tape.gradient(loss_num, model.trainable_weights)
    optim.apply_gradients(zip(grad, model.trainable_weights))

    print(output.shape)


# import sys
#
# sys.path.append('D:\\algorithm\\ocr')
# sys.path.append('E:\\algorithm\\ocr')
# import tensorflow as tf
# from tensorflow.keras import layers
# import tensorflow.keras as keras
# import numpy as np
#
# from Module.transformer import spatial_transformer_network as transformer
#
#
# def func(x):
#     b = np.min((30, 200))
#     for i in range(x.shape[1]):
#         x += 1
#     return x+b
#
#
# class RotateMyLayer(layers.Layer):
#     def __init__(self):
#         super(RotateMyLayer, self).__init__()
#
#     def call(self, shared_features,  box_info):
#         converted_f = tf.autograph.to_graph(func)
#         a = converted_f(box_info)
#         return box_info
#
#
# class RotateModel(keras.Model):
#     def __init__(self):
#         super(RotateModel, self).__init__()
#         self.layer = RotateMyLayer()
#
#     def call(self, shared_features, input_box_info):
#         a = self.layer(shared_features, input_box_info)
#         return a
#
#     def model(self):
#         input1 = keras.Input(shape=(None, None, 32), dtype=tf.float32)
#         # input2 = keras.Input(shape=[None, 6], dtype=tf.float32)
#         input3 = keras.Input(shape=(2, ), dtype=tf.int32)
#         return keras.Model(inputs=[input1, input3],
#                            outputs=self.call(input1, input3))
#
#
# if __name__ == '__main__':
#     # (3, 8, 384, 32)
#
#     shared_features = tf.random.normal([2, 112, 112, 32])
#     input_transform_matrix = tf.random.normal([3, 6])
#     input_box_masks = tf.expand_dims(tf.convert_to_tensor([2, 2, 1]), axis=0)
#     input_box_widths = tf.expand_dims(tf.convert_to_tensor([55, 12, 13]), axis=0)
#     input_box_info = tf.transpose(tf.concat([input_box_masks, input_box_widths], axis=0))
#
#     model = RotateModel().model()
#
#     output = model([shared_features, input_box_widths, input_box_info])
#     # print(output.shape)
