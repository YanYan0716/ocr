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
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self.detectmodel(x)


    def test_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self.detectmodel(x)