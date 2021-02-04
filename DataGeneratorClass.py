import itertools
import tensorflow as tf
import numpy as np
import cv2

class DataGen:
    '''
    如何使用类来构建数据库，但是在程序中没有使用这种方法
    '''
    def __init__(self, image_list, batch_size=4):
        self.i = 0
        self.image_list = image_list
        self.batch_size = batch_size

    def __load__(self, files_name):
        img = cv2.imread(files_name)
        return img

    def getitem(self, index):
        _img = self.__load__(self.image_list[index])

        return _img, self.image_list[index]

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < self.batch_size:
            img_arr, name = self.getitem(self.i)
            self.i += 1
        else:
            raise StopIteration()
        return img_arr, name

    def __call__(self):
        self.i = 0
        return self




if __name__ == '__main__':
    img_list = open('./icdar/train/images.txt', 'r').readlines()
    img_list = [img_mem.strip() for img_mem in img_list]
    gt_list = open('./icdar/train/GT.txt', 'r').readlines()
    gt_list = [gt_mem.strip() for gt_mem in gt_list]

    Generator = DataGen(img_list)
    # a=Generator.__next__()
    # print(a.shape)
    dataset = tf.data.Dataset.from_generator(
        Generator,
        (tf.int32, tf.string),
    )

    # for j in range(2):
    #     print('---------------------')
    #     for i in dataset:
    #         print(i.shape)

    dataset = dataset.batch(2).shuffle(buffer_size=1)
    for i in dataset:
        print(i[1])
