import itertools
import tensorflow as tf
import numpy as np
import cv2


def generator():
    img_list = open('./icdar/train/images.txt', 'r').readlines()
    batch_size = 4
    img_list = [img_mem.strip() for img_mem in img_list]
    img_list = np.array(img_list)
    index = np.arange(0, img_list.shape[0])

    if True:
        images = []
        for i in index:
            try:
                # 读取图片 对一张图片的处理
                img_fn = img_list[i]
                img = 1  # cv2.imread(img_fn)
                images.append(img)
                if len(images) == batch_size:
                    # print(len(images))
                    yield images
                images = []
            except:
                print('data reading have something error in DataGenerator.generator')
                break
                # continue
        # break


class DataGen:
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
