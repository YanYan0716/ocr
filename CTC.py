import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore", category=Warning)
import tensorflow as tf


if __name__ == '__main__':
    # 假设label_dict={’a':1, 'b':2}
    # 注意：在tf.nn.ctc_loss中不需要做softmax，因为本身包含这一过程，
    # 所以如果想笔算loss是否正确应该用下文中a的概率进行计算 其他地方和论文中计算结果相同
    # 参考资料：https://towardsdatascience.com/intuitively-understanding-connectionist-temporal-classification-3797e43a86c
    label = [1,2]
    logits = [[0, 0, 1], [0, 1, 0]]

    label_tensor = tf.convert_to_tensor([label], dtype=tf.int32)
    logits_tensor = tf.convert_to_tensor([logits], dtype=tf.float32)
    print(logits_tensor.shape)
    print(label_tensor.shape)

    a=tf.nn.softmax(logits_tensor)
    print(a)

    label_length = 2
    logits_length = 3
    labels_length_tensor = tf.convert_to_tensor([label_length], dtype=tf.int32)
    logits_length_tensor = tf.convert_to_tensor([logits_length], dtype=tf.int32)
    # logit_length = tf.fill([tf.shape(logits_tensor)[0]], tf.shape(logits_tensor)[1])
    print(logits_length_tensor.shape)

    '''
    tf.nn.ctc_loss: label_length:由于在每个batch中每个label的长度不是一样的，因此记录了每条数据的label的长度
                    logit_length:每个time step中的维数，有时候和num_classes是一样的
                    blank_index:空白的索引，此处空白并不是指空格符号，而是用来确定某个字符是否输出结束的
    label的第二种实现形式可以是稀疏矩阵，具体实现将在实际代码中
    '''
    loss = tf.nn.ctc_loss(label_tensor, logits_tensor, label_length=labels_length_tensor, logit_length= logits_length_tensor,
                          logits_time_major=False, blank_index=-1)
    print(loss.numpy())