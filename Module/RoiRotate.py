import numpy as np
import cv2
import tensorflow as tf
import math


class RoiRotate(object):
    def __init__(self, height=8):
        self.height = height

    def roi_rotate_tensor_while(self, feature_map, transform_matrixs, box_masks, box_width, is_debug = False):
        box_masks = tf.concat(box_masks, axis=0)
        box_nums = tf.shape(box_width)[0]

        max_width = 384

