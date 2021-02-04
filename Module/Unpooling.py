import sys

sys.path.append('D:\\algorithm\\ocr')
sys.path.append('E:\\algorithm\\ocr')
sys.path.append('/content/ocr/')
import tensorflow as tf
import numpy as np


@tf.function
def get_bilinear_filter(filter_shape, upscale_factor):
    kernel_size = 2 * upscale_factor - upscale_factor % 2
    if kernel_size % 2 == 1:
        centre_location = upscale_factor - 1
    else:
        center_location = upscale_factor - 0.5
    bilinear = np.zeros([filter_shape[0], filter_shape[1]])
    for x in range(filter_shape[0]):
        for y in range(filter_shape[1]):
            value = (1 - abs((x - center_location) / upscale_factor)) * (
                        1 - abs((y - center_location) / upscale_factor))
            bilinear[x, y] = value
    return bilinear


if __name__ == '__main__':
    bilinear = get_bilinear_filter([4, 4, 1, 1], 2)
    shared_features = tf.random.normal([1, 3, 6, 1])
    weights = tf.reshape(bilinear, [4, 4, 1, 1])
    # print(weights)
    stride = 2
    strides = [1, stride, stride, 1]
    deconv = tf.nn.conv2d_transpose(shared_features, weights, [1, 6, 12, 1], strides, padding='SAME')



'''
class UnpoolingLayer(layers.Layer):
    def __init__(self):
        super(UnpoolingLayer, self).__init__(name='unpooling')

    def call(self, x):
        input_shape = tf.shape(x)
        output_shape = input_shape
        shared_features = x[0]
        transform_matrix = x[1]
        box_info = x[2]
        box_nums = tf.shape(transform_matrix)[0]
        box_masks = box_info[:, 0]
        box_widths = box_info[:, 1]
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


class RotateModel(keras.Model):
    def __init__(self):
        super(RotateModel, self).__init__()
        self.layer = UnpoolingLayer()

    def call(self, x):
        a = self.layer(x)
        return a

    def model(self):
        input1 = keras.Input(shape=(None, None, 32), dtype=tf.float32)
        input2 = keras.Input(shape=(None, 6), dtype=tf.float32)
        input3 = keras.Input(shape=(2,), dtype=tf.int32)
        return keras.Model(inputs=[input1, input2, input3],
                           outputs=self.call([input1, input2, input3]))


if __name__ == '__main__':
    shared_features = tf.random.normal([2, 28, 28, 2])
'''
