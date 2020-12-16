import itertools
import tensorflow as tf


def gen():
    m=8
    for i in itertools.count(1):
        yield i


if __name__ == '__main__':
    # a=gen(5)
    dataset = tf.data.Dataset.from_generator(
        gen,
        (tf.int64),
    )

    for j in (dataset.take(6)):
        print(j)
