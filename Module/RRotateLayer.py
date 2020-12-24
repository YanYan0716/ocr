import sys

sys.path.append('D:\\algorithm\\ocr')
sys.path.append('E:\\algorithm\\ocr')
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as keras
import numpy as np

from Module.transformer import spatial_transformer_network as transformer


class RotateMyLayer(layers.Layer):
    def __init__(self):
        super(RotateMyLayer, self).__init__()

    def call(self, feature_map, transform_matrixs, box_masks, box_widths):
        box_num = tf.shape(box_widths)[0]
        # print(box_num)
        rois = []
        max_width = 384
        max_roi_num = 10
        for i in range(tf.minimum(max_roi_num, box_num)):
            index = box_masks[i]
            matrix = transform_matrixs[i]
            _feature_map = feature_map[tf.cast(index, tf.int32)]
            ex_feature_maps = tf.expand_dims(_feature_map, 0)
            trans_matrix = tf.expand_dims(matrix, 0)
            trans_feature_map = transformer(ex_feature_maps, trans_matrix, [8, box_widths[i]])
            roi = tf.image.crop_and_resize(trans_feature_map,  # 将每个trans_feature_map resize到[8, box_width]
                                           [[0.0, 0.0, 1.0, 1.0]],
                                           [0],
                                           [8, box_widths[i]],
                                           method='nearest',
                                           name='crop')
            # print(f'roi shape: {roi.shape}')
            pad_roi = tf.image.pad_to_bounding_box(roi,  # 将每个roi resize到[8, 384]
                                                   0,
                                                   0,
                                                   8,
                                                   max_width)

            rois.append(pad_roi)
        # qwe = tf.convert_to_tensor(np.array(rois, dtype=np.float))
        print('***************')
        # return tf.squeeze(rois, axis=1)
        # # print('ok')
        return feature_map


class mymodel(keras.Model):
    def __init__(self):
        super(mymodel, self).__init__()
        self.layer = RotateMyLayer()

    def call(self, shared_features, input_transform_matrix, input_box_masks, input_box_widths):
        a = self.layer(shared_features, input_transform_matrix, input_box_masks, input_box_widths)
        return a

    def model(self):
        input1 = keras.Input(shape=[None, None, 32])
        input2 = keras.Input(shape=[None, 6])
        input3 = keras.Input(shape=[None])
        input4 = keras.Input(shape=[None])
        return keras.Model((input1, input2, input3, input4), self.call(input1, input2, input3, input4))


if __name__ == '__main__':
    # (3, 8, 384, 32)

    shared_features = tf.random.normal([2, 112, 112, 32])
    input_transform_matrix = tf.random.normal([3, 6])
    input_box_masks = tf.convert_to_tensor([0, 0, 1])
    input_box_widths = tf.convert_to_tensor([55, 12, 13])
    model = mymodel()

    output = model(shared_features, input_transform_matrix, input_box_masks, input_box_widths)
    print(output.shape)
