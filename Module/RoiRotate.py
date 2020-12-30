import sys

sys.path.append('D:\\algorithm\\ocr')
sys.path.append('E:\\algorithm\\ocr')
sys.path.append('/content/ocr/')
import tensorflow as tf
import numpy as np

from Module.transformer import spatial_transformer_network as transformer


class RoiRotate(object):
    def __init__(self, height=8):
        self.height = height

    def roi_rotate_tensor_while(self, feature_map, transform_matrixs, box_masks, box_widths, is_debug=False):
        box_masks = tf.concat(box_masks, axis=0)
        box_nums = tf.shape(box_widths)[0]
        pad_rois = tf.TensorArray(tf.float32, box_nums)
        max_width = 384
        i = 0

        def cond(pad_rois, i):
            return i < box_nums

        def body(pad_rois, i):
            index = box_masks[i]
            matrix = transform_matrixs[i]
            _feature_map = feature_map[index]
            box_width = box_widths[i]

            ex_feature_maps = tf.expand_dims(_feature_map, 0)
            trans_matrix = tf.expand_dims(matrix, 0)
            # print(f'ex_feature_maps shape: {ex_feature_maps.shape}')
            trans_feature_map = transformer(ex_feature_maps, trans_matrix, [8, box_width])
            # print(f'trans_feature_map shape: {trans_feature_map.shape}')
            roi = tf.image.crop_and_resize(trans_feature_map,  # 将每个trans_feature_map resize到[8, box_width]
                                           [[0.0, 0.0, 1.0, 1.0]],
                                           [0],
                                           [8, box_width],
                                           method='nearest',
                                           name='crop')
            # print(f'roi shape: {roi.shape}')
            pad_roi = tf.image.pad_to_bounding_box(roi,  # 将每个roi resize到[8, 384]
                                                   0,
                                                   0,
                                                   8,
                                                   max_width)
            # print(f'pad_roi shape: {pad_roi.shape}')
            pad_rois = pad_rois.write(tf.cast(i, tf.int32), pad_roi)
            i += 1
            return pad_rois, i

        pad_rois, _ = tf.while_loop(cond, body, loop_vars=[pad_rois, i], parallel_iterations=10, swap_memory=True)
        pad_rois = pad_rois.stack()
        return tf.squeeze(pad_rois, axis=1)


if __name__ == '__main__':
    shared_features = tf.random.normal([2, 112, 112, 32])
    input_transform_matrix = tf.random.normal([3, 6])
    input_box_masks = [np.array([0, 0]), np.array([1])]
    input_box_widths = [55, 12, 13]
    ROI = RoiRotate()
    output = ROI.roi_rotate_tensor_while(shared_features,
                                         input_transform_matrix,
                                         input_box_masks,
                                         input_box_widths)
    print(output.shape)
