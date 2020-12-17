import tensorflow as tf

import Module.transformer.spatial_transformer_network as transformer


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
            trans_feature_map = transformer(ex_feature_maps, trans_matrix, [8, box_width])
            roi = tf.image.crop_and_resize(trans_feature_map,
                                           [[0.0, 0.0, 1.0, 1.0]],
                                           [0],
                                           [8, box_width],
                                           method='nearest',
                                           name='crop')
            pad_roi = tf.image.pad_to_bounding_box(roi,
                                                   0,
                                                   0,
                                                   8,
                                                   max_width)
            pad_rois = pad_rois.write(tf.cast(i, tf.int32), pad_roi)
            i += 1
            return pad_rois, i

        pad_rois, _ = tf.while_loop(cond, body, loop_vars=[pad_rois, i], parallel_iterations=10, swap_memory=True)
        return pad_rois.stack()


if __name__ == '__main__':
        pass
