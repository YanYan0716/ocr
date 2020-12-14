'''
训练数据的生成
'''
import numpy as np
import cv2
import tensorflow as tf
import pandas as pd

import config


def label_to_array(label):
    '''
    将实际的字符转为数字标签
    :param label:
    :return:
    '''
    try:
        label = label.replace(' ', '')
        return [config.CHAR_VECTOR.index(x) for x in label]
    except:
        print('something error in DaraGenerator.label_to_array')
    # return label


def load_annoatation(p):
    '''
    加载gt.txt中的内容，变为训练过程中的输入格式
    :param p:
    :return:
    '''
    text_polys = []  # 存储矩形框
    text_tags = []  # 存储是否有字符，有为False 没有为True
    labels = []  # 存储字符的内容，此处已经将字母转换为对应的数字label
    with open(p, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.replace('\xef\xbb\xbf', '')
            line = line.replace('\xe2\x80\x8d', '')
            line = line.strip()
            line = line.split(',')
            if len(line) > 9:
                label = line[8]
                for i in range(len(line) - 9):
                    label = label + ',' + line[i + 9].lower()
            else:
                label = line[-1].lower()
            label = label.lstrip()  # 去掉左边的特殊字符
            temp_line = list(map(eval, line[:8]))
            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, temp_line)
            # 获得文字的矩形框
            cnt = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)  # box.shape (4, 2)
            text_polys.append(box)
            if label == '*' or label == '###' or label == '':
                text_tags.append(True)
                labels.append([-1])
            else:
                labels.append(label_to_array(label))
    return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool), labels


def polygon_area(poly):
    '''
    计算面积
    :param poly:
    :return:
    '''
    edge = [
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1]),
    ]
    return np.sum(edge) / 2


def check_and_validate_polys(polys, tags, height, width):
    '''
    检查获取到的矩形框四个顶点的顺序对不对，过滤掉面积为0的情况
    :param polys:
    :param tags:
    :param height:
    :param width:
    :return:
    '''
    if polys.shape[0] == 0:
        return polys
    # 矩形框的坐标应该在整张图片的范围里
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, width - 1)
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, height - 1)

    validated_polys = []
    validated_tags = []

    for poly, tag in zip(polys, tags):
        p_area = polygon_area(poly)  # 计算矩形框的面积
        if abs(p_area) < 1:
            continue
        if p_area > 0:  # 保证四个顶点的顺序是左上，右上，右下， 左下
            poly = poly[(0, 3, 2, 1), :]
        validated_polys.append(poly)
        validated_tags.append(tag)
    return np.array(validated_polys), np.array(validated_tags)


def generator(img_list=None,
              gt_list=None,
              input_size=512,
              batch_size=32,
              bachground_ratio=0,
              random_scale=np.array([0.5, 0.6, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.4, 1.6, 2, 3, 4])
              ):
    '''
    原文位置：icdar.py line:823
    :param gt_list:
    :param img_list:
    :param input_size:
    :param batch_size:
    :param bachground_ratio:
    :param random_scale:
    :return:
    '''
    max_box_num = 64
    max_box_width = 384
    img_list = np.array(img_list)
    gt_list = np.array(gt_list)
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
            try:
                # 读取图片
                img_fn = img_list[i]
                img = cv2.imread(img_fn)
                if img is None:
                    continue
                h, w, _ = img.shape
                # 读取图片对应的GT.txt文件
                txt_fn = gt_list[i]

                # 返回的数据类型：np.array, np.array, list
                text_polys, text_tags, text_label = load_annoatation(txt_fn)
                # 检查获取到的矩形框四个顶点的顺序对不对，过滤掉面积为0的情况
                text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, h, w)
                rd_scale = np.random.choice(random_scale) # 随机选一个scale
                img = cv2.resize(img, dsize=None, fx=rd_scale, fy=rd_scale)
                file_name =0
            except:
                print('data reading have something error in DataGenerator.generator')
            break
        break


if __name__ == '__main__':
    df = pd.read_csv('./icdar/train/train.csv')
    file_paths = df['file_name'].values
    gt_paths = df['gt'].values
    ds_train = tf.data.Dataset.from_tensor_slices((file_paths, gt_paths))
    # for i in ds_train:
    #     print(i)
    # ds_train = tf.data.Dataset.from_generator()
    # img_list = open('./icdar/train/images.txt', 'r').readlines()
    # img_list = [img_mem.strip() for img_mem in img_list]
    # gt_list = open('./icdar/train/GT.txt', 'r').readlines()
    # gt_list = [gt_mem.strip() for gt_mem in gt_list]
    # generator(img_list, gt_list)