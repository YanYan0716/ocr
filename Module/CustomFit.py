import tensorflow as tf
import tensorflow.keras as keras


class CustomFit(keras.Model):
    def __init__(self, detectmodel, roiroteta, regmodel):
        super(CustomFit, self).__init__()
        self.detectmodel = detectmodel
        self.roirotate = roiroteta
        self.regmodel = regmodel

    def compile(self, detectloss, regloss, optimizer='rmsprop',):
        super(CustomFit, self).compile()
        self.optimizer = optimizer
        self.detectloss = detectloss
        self.regloss = regloss

    def train_step(self, data):
        images, image_fns, score_maps, geo_maps, training_masks, transform_matrixes, boxes_masks, box_widths, \
        text_labels_sparse_0, text_labels_sparse_1, text_labels_sparse_2 = data

        # with tf.GradientTape() as tape:
        #     y_pred = self.detectmodel(x)
