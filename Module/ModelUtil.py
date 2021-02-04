import tensorflow as tf


def computer():
    batch_size = 4
    batch_data = tf.TensorArray(tf.float32, batch_size)

    def body(i, batch_data):
        input_data = tf.random.normal([32, 32, 3])
        # print(f'---- {input_data}')
        batch_data = batch_data.write(tf.cast(i, tf.int32), input_data)
        i += 1
        return i, batch_data

    def cond(i, batch_data):
        return i < batch_size

    _, batch_data = tf.while_loop(cond, body, [0, batch_data])
    return batch_data.stack()


if __name__ == '__main__':
    text = computer()
    print('*********************************')
    print(text.shape)