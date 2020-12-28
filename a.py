import tensorflow as tf


@tf.function
def func(a, b):
    return a+b


if __name__ == '__main__':
    a = tf.constant(value=3)
    b = tf.constant(value=2)

    c = tf.register_tensor_conversion_function((tf.float32, tf.float32), func)
    print(c)


https://www.kite.com/python/docs/keras.backend.moving_averages.distribution_strategy_context.distribute_lib.losses_impl.nn.rnn_cell.tf_utils.register_symbolic_tensor_type