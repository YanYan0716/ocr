'''
训练数据的生成
'''
import numpy as np


def generator(img_list=None,
              input_size=512,
              batch_size=32,
              bachground_ratio=0,
              random_scale=np.array([0.5, 0.6, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.4, 1.6, 2, 3, 4])
              ):
    max_box_num = 64
    max_box_width = 384
    img_list = np.array(img_list)
    index = np.arange(0, img_list.shape[0])

    while True:
        # 打乱数据
        np.random.shuffle(index)
        np.random.shuffle(index)

        images = []
        image_fns = []
        score_maps = []
        geo_maps = []
        training_masks = []

        test_ployses = []
        text_tagses = []
        boxes_masks = []
        text_labels = []
        count = 0
        for i in index:
            print(i)
        break
    pass


if __name__ == '__main__':
    img_path = './icdar/train/images.txt'
    img_list = open(img_path, 'r').readlines()
    generator(img_list)
