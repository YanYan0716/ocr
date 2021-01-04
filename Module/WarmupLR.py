import tensorflow as tf
import tensorflow.keras as keras

from RRotateLayer import RotateModel


class WarmUpLR(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, final_LR, d_model, warmup_steps=800, warm_time=100):
        super(WarmUpLR, self).__init__()
        self.final_LR = final_LR
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_step = warmup_steps
        self.warm_time = warm_time

    def __call__(self, step):
        if step < self.warm_time:
            step = tf.cast(step, tf.float32)
            arg1 = tf.math.rsqrt(step)
            arg2 = step * (self.warmup_step ** -1.5)
            lr = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
        elif (step >= self.warm_time) and (step < (self.final_LR - 20)):
            lr = self.final_LR
        else:
            lr = self.final_LR * 0.1
        return lr


if __name__ == '__main__':
    shared_features = tf.random.normal([2, 112, 112, 32])
    input_transform_matrix = tf.random.normal([3, 6])
    input_box_masks = tf.expand_dims(tf.convert_to_tensor([0, 0, 1]), axis=0)
    input_box_widths = tf.expand_dims(tf.convert_to_tensor([55, 12, 13]), axis=0)
    input_box_info = tf.transpose(tf.concat([input_box_masks, input_box_widths], axis=0))
    # print(input_box_info.shape)

    learning_rate = WarmUpLR(128)
    model = keras.models.Sequential([tf.keras.layers.Dense(10)])
    optim = keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98)

    model = RotateModel().model()

    for i in range(100):
        with tf.GradientTape() as tape:
            output = model([shared_features, input_transform_matrix, input_box_info])
        total_loss = tf.reduce_mean(output)

        optim.learning_rate = WarmUpLR(i)
        grad = tape.gradient([total_loss], model.trainable_weights)
        optim.apply_gradients(zip(grad, model.trainable_weights))
        # print(optim.learning_rate.__call__(i))
