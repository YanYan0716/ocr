import sys

sys.path.append('D:\\algorithm\\ocr')
sys.path.append('E:\\algorithm\\ocr')
sys.path.append('/content/ocr/')
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
        # j = 0
        # for i in range(box_nums):
        #     index = box_masks[i]
        #     matrix = transform_matrix[i]
        #     feature_map = shared_features[index]
        #     box_width = box_widths[i]
        #     ex_feature_maps = tf.expand_dims(feature_map, 0)
        #     trans_matrix = tf.expand_dims(matrix, 0)
        #     pad_rois = pad_rois.write(j, trans_matrix)
        #     j += 1
        #     pad_rois = pad_rois.write(j, trans_matrix)
        #     j +=1
            # pad_rois = pad_rois.write(i, trans_matrix)
    # pad_rois = tf.TensorArray(tf.float32, dynamic_size=True)
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
        super(RotateMyLayer, self).__init__(name='roitate')

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
        return a

    def model(self):
        input1 = keras.Input(shape=(None, None, 32), dtype=tf.float32)
        input2 = keras.Input(shape=(None, 6), dtype=tf.float32)
        input3 = keras.Input(shape=(2, ), dtype=tf.int32)
        return keras.Model(inputs=[input1,input2,input3],
                           outputs=self.call([input1, input2, input3]))


if __name__ == '__main__':
    # (3, 8, 384, 32)

    shared_features = tf.random.normal([2, 112, 112, 32])
    input_transform_matrix = tf.random.normal([3, 6])
    input_box_masks = tf.expand_dims(tf.convert_to_tensor([0, 0, 1]), axis=0)
    input_box_widths = tf.expand_dims(tf.convert_to_tensor([55, 12, 13]), axis=0)
    input_box_info = tf.transpose(tf.concat([input_box_masks, input_box_widths], axis=0))
    print(input_box_info.shape)

    model = RotateModel().model()

    output = model([shared_features, input_transform_matrix,  input_box_info])
    print(output.shape)
