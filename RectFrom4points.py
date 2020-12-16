import itertools
import tensorflow as tf


def gen(m):
    m=8
    for i in itertools.count(1):
        yield m, [1] * m


if __name__ == '__main__':
    a=gen(5)
    dataset = tf.data.Dataset.from_generator(
        a,
        (tf.int64, tf.int64),
        # (tf.TensorShape([]), tf.TensorShape([None]))
    )

    # list(dataset.take(3).as_numpy_iterator())
    for j in (dataset.take(3)):
        print(j)
    # print(list(dataset.take(3).as_numpy_iterator()))
