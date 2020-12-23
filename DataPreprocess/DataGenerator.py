'''
训练数据的生成
'''
import math
import sys
from itertools import compress

sys.path.append('D:\\algorithm\\ocr')
sys.path.append('E:\\algorithm\\ocr')
import numpy as np
import cv2
import tensorflow as tf
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
import PIL.Image as Image
import config
import Aug_Operations as aug

from DataPreprocess.PrepareForGRB import shrink_poly, earn_rect_angle, point_dist_to_line


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
            line = line.replace('\ufeff', '')
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
            cnt = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)  # box.shape (4, 2)
            text_polys.append(box)
            if label == '*' or label == '###':  # or label == '':
                text_tags.append(True)
                labels.append([-1])
            else:
                labels.append(label_to_array(label))
                text_tags.append(False)
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


def img_aug(img, file_name='./temporary.jpg'):
    ratio = random.uniform(0, 1)
    if ratio < 0.1:
        img = cv2.blur(img, (3, 3))

    if ratio < 0.2:  # 对每个像素点加减像素值作为augment
        w, h, c = img.shape
        noise_img = np.random.randint(0, 30, size=(w, h, c)) - 15
        timg = noise_img + img
        timg = np.minimum(timg, 255)
        timg = np.maximum(timg, 0)
        img = timg.astype(np.uint8)

    if ratio < 0.3:  # 随机交换颜色通道来改变像素值
        index = [0, 1, 2]
        random.shuffle(index)
        a = img[:, :, index[0]]
        b = img[:, :, index[1]]
        c = img[:, :, index[2]]
        img = cv2.merge([a, b, c])

    elif ratio < 0.4:  # 按照com_ration的数值作为保存的图像质量参数
        com_ratio = random.randint(1, 20)
        cv2.imwrite(file_name, img, [int(cv2.IMWRITE_JPEG_QUALITY), com_ratio])
        img = cv2.imread(file_name)
        os.remove(file_name)

    if random.uniform(0, 1) < 0.3:
        k1 = random.randint(10, 30)
        k2 = random.randint(1, 5)
        dis = aug.Distort(1.0, k1, k1, k2)  # 数据扩展 没仔细看
        img_a = dis.perform_operation(Image.fromarray(img))
        img = np.array(img_a)

    ratio_2 = random.uniform(0, 1)
    if ratio_2 < 0.5:
        img = 255 - img
    return img


def crop_area(img, polys, tags, crop_background=False, max_tries=50):
    h, w, _ = img.shape
    pad_h = h // 10  # 除法取整
    pad_w = w // 10
    h_array = np.zeros((h + pad_h * 2), dtype=np.int32)
    w_array = np.zeros((w + pad_w * 2), dtype=np.int32)
    for poly in polys:
        poly = np.round(poly, decimals=0).astype(np.int32)  # 将矩形框的坐标转为整数
        min_x = np.min(poly[:, 0])
        max_x = np.max(poly[:, 0])
        w_array[min_x + pad_w:max_x + pad_w] = 1  # 相当于找到边缘扩充后的矩形框位置
        min_y = np.min(poly[:, 1])
        max_y = np.max(poly[:, 1])
        h_array[min_y + pad_h:max_y + pad_h] = 1

    # 判断是否存在不在矩形框中的点，如果不存在就说明整张图都在矩形框里，直接返回结果
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]
    if len(h_axis) == 0 or len(w_axis) == 0:
        return img, polys, tags, np.array(len(polys))

    for i in range(max_tries):
        xx = np.random.choice(w_axis, size=2)
        xmin = np.min(xx) - pad_w
        xmax = np.max(xx) - pad_w
        xmin = np.clip(xmin, 0, w - 1)
        xmax = np.clip(xmax, 0, w - 1)
        yy = np.random.choice(h_axis, size=2)
        ymin = np.min(yy) - pad_h
        ymax = np.max(yy) - pad_w
        ymin = np.clip(ymin, 0, h - 1)
        ymax = np.clip(ymax, 0, h - 1)
        # 过滤掉剪裁面积过小的情况
        if xmax - xmin < 0.1 * w or ymax - ymin < 0.1 * h:
            continue
        if polys.shape[0] != 0:
            poly_axis_in_area = (polys[:, :, 0] >= xmin) & \
                                (polys[:, :, 0] <= xmax) & \
                                (polys[:, :, 1] >= ymin) & \
                                (polys[:, :, 1] <= ymax)
            selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]
        else:
            selected_polys = []
        if len(selected_polys) == 0:
            if crop_background:
                return img[ymin:ymax + 1, xmin:xmax + 1, :], polys[selected_polys], tags[selected_polys], selected_polys
            else:
                continue

        img = img[ymin: ymax + 1, xmin: xmax + 1, :]
        polys = polys[selected_polys]
        tags = tags[selected_polys]
        polys[:, :, 0] -= xmin
        polys[:, :, 1] -= ymin
        return img, polys, tags, selected_polys
    return img, polys, tags, np.array(range(len(polys)))


def generate_rbox(img_size, polys, tags):
    '''
    吧任意的四边形转换为包含这四个顶点的最小外接矩形
    :param img_size: 
    :param polys: 
    :param tags: 
    :return: 
    '''
    h, w = img_size
    poly_mask = np.zeros((h, w), dtype=np.uint8)  # 掩膜，类似于image segmentation中，岁每个像素进行分类
    score_map = np.zeros((h, w), dtype=np.uint8)
    geo_map = np.zeros((h, w, 5), dtype=np.float32)

    training_mask = np.ones((h, w), dtype=np.uint8)  # 标志图像中该点的像素值是否接受训练，数值为0表示不接受训练
    rectangles = []

    for poly_idx, poly_tag in enumerate(zip(polys, tags)):
        poly = poly_tag[0]
        tag = poly_tag[1]

        r = [None, None, None, None]
        for i in range(4):
            r[i] = min(np.linalg.norm(poly[i] - poly[(i + 1) % 4]),
                       np.linalg.norm(poly[i] - poly[(i - 1) % 4]))

        shrinked_poly = shrink_poly(poly.copy(), r).astype(np.int32)[np.newaxis, :, :]
        # 生成score_map 在shrinked_poly的范围内的像素点位置的值为1 否则为0
        cv2.fillPoly(score_map, shrinked_poly, 1)
        # 标记poly_mask 相当于对像素点打标签，0为背景
        cv2.fillPoly(poly_mask, shrinked_poly, poly_idx + 1)
        # 如果找到的矩形区域过小，那么就不再用
        poly_h = min(np.linalg.norm(poly[0] - poly[3]), np.linalg.norm(poly[1] - poly[2]))
        poly_w = min(np.linalg.norm(poly[0] - poly[1]), np.linalg.norm(poly[2] - poly[3]))
        if min(poly_h, poly_w) < config.MIN_TEXT_SIZE:
            cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
        if tag:
            cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)

        # 获得该矩形框内像素点的坐标位置
        xy_in_poly = np.argwhere(poly_mask == (poly_idx + 1))

        # 获得最小外接矩形和旋转角
        rectangle, angle = earn_rect_angle(poly)
        rectangles.append(rectangle.flatten())
        p0_rect, p1_rect, p2_rect, p3_rect = rectangle

        # 求矩形框中的每个像素点到四条边界的距离
        for x, y in xy_in_poly:
            point = np.array([x, y], dtype=np.float32)
            # top
            geo_map[y, x, 0] = point_dist_to_line(p0_rect, p1_rect, point)
            # right
            geo_map[y, x, 1] = point_dist_to_line(p1_rect, p2_rect, point)
            # bottom
            geo_map[y, x, 2] = point_dist_to_line(p2_rect, p3_rect, point)
            # left
            geo_map[y, x, 3] = point_dist_to_line(p3_rect, p0_rect, point)
            geo_map[y, x, 4] = angle

    return score_map, geo_map, training_mask, rectangles


def distance_2p(point1, point2):
    '''
    计算两点之间的距离
    :param point1:
    :param point2:
    :return:
    '''
    return np.sqrt(np.sum((point1 - point2) * (point1 - point2)))


def get_project_matrix_and_width(polys):
    '''
    计算一个标准矩形和polys之间的仿射矩阵
    :param polys:
    :return:
    '''
    project_matrixes = []
    box_widths = []

    for i in range(polys.shape[0]):  # 矩形框的个数
        x1, y1, x2, y2, x3, y3, x4, y4 = polys[i] / 2  # 这里的除以2和送入RoIRotate的特征图尺寸有关系，应该是一致的
        width = distance_2p(np.array([x1, y1]), np.array([x2, y2]))
        height = distance_2p(np.array([x1, y1]), np.array([x4, y4]))

        # 两种斜体增强方式，通过改变两条对角线的横坐标
        ratio = random.uniform(0, 1)
        if ratio < 0.3 and width > 2 * height:
            delta = 0.5 * height
            x1 = x1 - delta
            x3 = x3 + delta
        elif ratio < 0.6 and width > 2 * height:
            delta = 0.5 * height
            x2 = x2 - delta
            x4 = x4 + delta

        delta1 = random.uniform(-0.1, 0.1)
        delta2 = random.uniform(-0.1, 0.1)
        delta3 = random.uniform(-0.3, 0.3)
        delta4 = random.uniform(-0.3, 0.3)

        if width > 1.5 * height:  # 宽大于长的情况
            # math.ceil:返回 >= 参数的最小整数 ？？？？？？？？？和ROIRotate有关，，，，，，
            width_box = math.ceil(8 * width / height)
            # 对于x1x2保持y坐标的一致，也就是平行于x轴做小范围的长度变化
            # 对于x1x4保持x坐标的一致，也就是平行于y轴做小范围的长度变化
            src_pts = np.float32([(x1 + delta3 * height, y1 + delta1 * height),
                                  (x2 + delta4 * height, y2 + delta1 * height),
                                  (x4 + delta3 * height, y4 + delta2 * height)])
        else:
            # 对于x2x3保持x坐标的一致，也就是平行于y轴做小范围的长度变化
            # 对于x1x2保持y坐标的一致，也就是平行于x轴做小范围的长度变化
            width_box = math.ceil(8 * height / width)
            src_pts = np.float32([(x2 + delta2 * width, y2 + delta3 * width),
                                  (x3 + delta2 * width, y3 + delta4 * width),
                                  (x1 + delta1 * width, y1 + delta3 * width)])
        mapped_x1, mapped_y1 = (0, 0)
        mapped_x4, mapped_y4 = (0, 8)
        mapped_x2, mapped_y2 = (width_box, 0)
        dst_pts = np.float32([(mapped_x1, mapped_y1),
                              (mapped_x2, mapped_y2),
                              (mapped_x4, mapped_y4)])
        # 得到从dst_pts到src_pts的仿射矩阵
        affine_matrix = cv2.getAffineTransform(dst_pts.astype(np.float32), src_pts.astype(np.float32))
        affine_matrix = affine_matrix.flatten()

        project_matrixes.append(affine_matrix)
        box_widths.append(width_box)
    project_matrixes = np.array(project_matrixes)
    box_widths = np.array(box_widths)

    return project_matrixes, box_widths


def sparse_tuple_from(sequences, dtype=np.int32):
    indices = []
    values = []
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), [i for i in range(len(seq))]))
        values.extend(seq)
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def generator(img_list=None,
              gt_list=None,
              INPUT_SIZE=512,
              batch_size=1,
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
    img_list = open('./icdar/train/images.txt', 'r').readlines()
    img_list = [img_mem.strip() for img_mem in img_list]
    gt_list = open('./icdar/train/GT.txt', 'r').readlines()
    gt_list = [gt_mem.strip() for gt_mem in gt_list]

    max_box_num = 64
    max_box_width = 384
    img_list = np.array(img_list)
    gt_list = np.array(gt_list)
    index = np.arange(0, img_list.shape[0])


    while True:
        # 打乱数据
        # np.random.shuffle(index)
        # np.random.shuffle(index)
        images = []
        image_fns = []
        score_maps = []
        geo_maps = []
        training_masks = []
        text_polyses = []
        text_tagses = []
        boxes_masks = []
        text_labels = []
        count = 0
        for i in index:
            try:
                # 读取图片 对一张图片的处理
                img_fn = img_list[i]
                img = cv2.imread(img_fn)
                if img is None:
                    continue
                h, w, _ = img.shape
                # 读取图片对应的GT.txt文件
                txt_fn = gt_list[i]

                # 返回的数据类型：np.array, np.array, list
                text_polys, text_tags, text_label = load_annoatation(txt_fn)
                # print('----------------------------------')
                # 检查获取到的矩形框四个顶点的顺序对不对，过滤掉面积为0的情况
                text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, h, w)
                # print(text_polys.shape, text_tags)
                rd_scale = np.random.choice(random_scale)  # 随机选一个scale

                img = cv2.resize(img, dsize=None, fx=rd_scale, fy=rd_scale)
                file_name = img_fn.split('/')[-1]
                img = img_aug(img, file_name)
                text_polys *= rd_scale

                # 对图片进行裁剪
                img, text_polys, text_tags, selected_poly = crop_area(img, text_polys, text_tags, crop_background=False)
                # print(text_polys.shape, text_tags)
                if text_polys.shape[0] == 0 or len(text_label) == 0:
                    continue
                h, w, _ = img.shape
                max_h_w_i = np.max([h, w, INPUT_SIZE])
                img_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                img_padded[:h, :w, :] = img.copy()  # 按照最长边进行padding，边长的最小值是INPUT_SIZE
                img = img_padded
                # 把图片resize到512的尺寸
                new_h, new_w, _ = img.shape
                img = cv2.resize(img, dsize=(INPUT_SIZE, INPUT_SIZE))
                resize_ratio_x = INPUT_SIZE / float(new_h)
                resize_ratio_y = INPUT_SIZE / float(new_w)
                text_polys[:, :, 0] *= resize_ratio_x
                text_polys[:, :, 1] *= resize_ratio_y
                new_h, new_w, _ = img.shape

                # 生成矩形框，这里的矩形是基于四个顶点坐标，带有旋转角度
                score_map, geo_map, training_mask, rectangles = generate_rbox((new_h, new_w), text_polys, text_tags)
                # print(rectangles[0])

                text_label = [text_label[i] for i in selected_poly]

                # 文字信息的掩码，如果这个位置有信息就是True, 否则为False
                mask = [not (word == [-1]) for word in text_label]

                # 过滤掉文字信息为False的标签和矩形框
                text_label = list(compress(text_label, mask))
                rectangles = list(compress(rectangles, mask))

                assert len(text_label) == len(rectangles)
                if len(text_label) == 0:
                    continue

                boxes_mask = [count] * len(rectangles)
                count += 1
                # 以上部分是对一张图片的处理

                # 把一张图片的信息加到一个batch中
                images.append(img[:, :, ::-1].astype(np.float32))
                image_fns.append(img_fn)
                # np.array([1,2,3,4,5,6,7,8,9,0])[::4] 表示每隔4个点取一个点，所以score_map shape变为【128， 128】
                score_maps.append(score_map[::4, ::4, np.newaxis])
                geo_maps.append(geo_map[::4, ::4, :].astype(np.float32))
                training_masks.append(training_mask[::4, ::4, np.newaxis].astype(np.float32))
                text_polyses.append(rectangles)
                boxes_masks.extend(boxes_mask)
                text_labels.extend(text_label)
                text_tagses.append(text_tags)  # 其中False的数量 = len(test_ployses[0])

                # 打印一个image的信息
                # print('--------------总结一个image的相关输出--------------')
                # print(f'images len: {len(images)}, \timage[0] shape: {images[0].shape}')
                # print(f'image_fns len: {len(image_fns)}, \timage_fns[0] : {image_fns[0]}')
                # print(f'score_maps len: {len(score_maps)}, \tscore_maps[0]  shape: {score_maps[0].shape}')
                # print(f'geo_maps len: {len(geo_maps)}, \tgeo_maps[0]  shape: {geo_maps[0].shape}')
                # print(f'training_masks len: {len(training_masks)}, \ttraining_masks[0]  shape: {training_masks[0].shape}')
                # print(f'text_polyses len: {len(text_polyses)}, \text_polyses[0][0] len: {len(text_polyses[0])}')#, \ttext_polys[0] len: {len(test_ployses[0])}')
                # print(f'boxes_masks len: {len(boxes_masks)}, \tboxes_masks[0]: {boxes_masks[0]}')
                # print(f'text_labels len: {len(text_labels)}, \ttext_labels[0]: {text_labels[0]}')
                # print(f'text_tagses len: {len(text_tagses)}, \ttext_tagses[0]: {text_tagses[0]}')
                
                # 如果图片的数量够一个batch_size
                if len(images) == batch_size:
                    text_polyses = np.concatenate(text_polyses)
                    text_tagses = np.concatenate(text_tagses)
                    transform_matrixes, box_widths = get_project_matrix_and_width(text_polyses)

                    # 一张图片中每一个矩形框都对应一个仿射矩阵，但是我们限制了一个batch_size中最多的矩形框个数不超过max_box_num
                    # 如果数量超过了，就随机sample max_box_num个矩形框
                    num_array = [ni for ni in range(len(transform_matrixes))]  # 所有矩形框的索引
                    if len(transform_matrixes) > max_box_num:
                        sub_array = random.sample(num_array, max_box_num)
                        sub_index = [int(ai in sub_array) for ai in num_array]
                        del_num = 0

                        # 删除在sample之外或者box_widths过大的矩形框的相关信息
                        for si in num_array:
                            if (sub_index[si] == 0) or (box_widths[si - del_num] > max_box_width):
                                transform_matrixes = np.delete(transform_matrixes, si - del_num, 0)
                                boxes_masks = np.delete(boxes_masks, si - del_num, 0)
                                box_widths = np.delete(box_widths, si - del_num, 0)
                                text_labels = np.delete(text_labels, si - del_num, 0)
                                del_num += 1
                    else:
                        del_num = 0
                        for si in num_array:
                            if box_widths[si - del_num] > max_box_width:
                                transform_matrixes = np.delete(transform_matrixes, si - del_num, 0)
                                boxes_masks = np.delete(boxes_masks, si - del_num, 0)
                                box_widths = np.delete(box_widths, si - del_num, 0)
                                text_labels = np.delete(text_labels, si - del_num, 0)
                                del_num += 1
                    # print(np.array(text_labels))
                    # print(len(transform_matrixes))
                    # text_labels_sparse:
                    #   1. 第一维是这个字符属于第几个矩形框，第二维是顺序索引，
                    #   2. 所有字符串中字符顺序排列后的数字index
                    #   3. 两个数字，前者表示一共有多少个字符串，后者是这些字符串中最常的字符串的len
                    text_labels_sparse = sparse_tuple_from(np.array(text_labels))

                    boxes_masks2 = []
                    for bi in range(batch_size):
                        cnt = sum([int(bi == bmi) for bmi in boxes_masks])
                        if cnt < 1:
                            boxes_masks2.append(np.array([]))
                        else:
                            boxes_masks2.append(np.array([bi] * cnt))
                    # boxes_masks长度为batch_size，每一个元素是一个array，长度为本张图片的矩形框个数，
                    # 填充的值为这张图片在batch_size中的索引，如果某张图片没有矩形框那么这个array就是空
                    boxes_masks = boxes_masks2
                    # 打印一个batch_size的信息
                    print('--------------总结一个batch_size的相关输出--------------')
                    print(f'images len: {len(images)}, \t\t\timage[0] shape: {images[0].shape}')
                    print(f'image_fns len: {len(image_fns)}, \t\timage_fns[0] : {image_fns[0]}')
                    print(f'score_maps len: {len(score_maps)}, \t\tscore_maps[0]  shape: {score_maps[0].shape}')
                    print(f'geo_maps len: {len(geo_maps)}, \t\tgeo_maps[0]  shape: {geo_maps[0].shape}')
                    print(
                        f'training_masks len: {len(training_masks)}, \t\ttraining_masks[0]  shape: {training_masks[0].shape}')
                    print(
                        f'transform_matrixes len: {len(transform_matrixes)}, \ttransform_matrixes[0][0] len: {len(transform_matrixes[0])}')
                    print(transform_matrixes)
                    print(f'boxes_masks len: {len(boxes_masks)}, \t\tboxes_masks: {boxes_masks}')
                    print(f'box_widths len: {len(box_widths)}, \t\tbox_widths: {box_widths}')
                    print(
                        f'text_labels_sparse len: {len(text_labels_sparse)}, \ttext_labels_sparse[0] shape: {text_labels_sparse[0].shape}')
                    print(f'--------------------------\ttext_labels_sparse[1] len: {len(text_labels_sparse[1])}')
                    print(f'--------------------------\ttext_labels_sparse[2]: {text_labels_sparse[2]}')
                    print('--------------------******************-------------------')
                    yield images, image_fns, score_maps, geo_maps, training_masks, transform_matrixes, boxes_masks, box_widths, text_labels_sparse
                    images = []
                    image_fns = []
                    score_maps = []
                    geo_maps = []
                    training_masks = []
                    text_polyses = []
                    text_tagses = []
                    boxes_masks = []
                    text_labels = []
                    count = 0
                    break
            except:
                print('data reading have something error in DataGenerator.generator')
                continue
        break


def read_img(img_path, gt_path):
    '''
    改写generator函数，以适应tf.data.Dataset.from_tensor_slices
    :param img_path:
    :param gt_path:
    :return:
    '''
    RANDOM_SCALE = np.array([0.5, 0.6, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.4, 1.6, 2, 3, 4])
    MAX_BOX_NUM = 64
    MAX_BOX_WIDTH = 384
    INPUT_SIZE = 512

    # 读取图片
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)  # tf和cv2的通道格式是不一样的

    img = cv2.cvtColor(img.numpy(), cv2.COLOR_RGB2BGR)
    h, w, _ = img.shape

    # 读取图片对应的GT.txt文件, 返回的数据类型：np.array, np.array, list
    # text_polys: 文字坐标框
    # text_tags: 标志位，有文字是False，没有是True
    # text_label: 文字内容
    text_polys, text_tags, text_label = load_annoatation(gt_path)

    # 检查获取到的矩形框四个顶点的顺序对不对，过滤掉面积为0的情况
    text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, h, w)

    # 对图片进行缩放、通道变换等操作 所以矩形框也要按照同比例缩放 此过程未涉及图像crop
    rd_scale = np.random.choice(RANDOM_SCALE)  # 随机选一个scale
    img = cv2.resize(img, dsize=None, fx=rd_scale, fy=rd_scale)
    img = img_aug(img)
    text_polys *= rd_scale

    # 对图片进行裁剪
    flag_num = 0
    img, text_polys, text_tags, selected_poly = crop_area(img, text_polys, text_tags, crop_background=False)
    while (text_polys.shape[0] == 0 or len(text_label) == 0) and flag_num < 50:
        flag_num += 1
        img, text_polys, text_tags, selected_poly = crop_area(img, text_polys, text_tags, crop_background=False)
    h, w, _ = img.shape
    max_h_w_i = np.max([h, w, INPUT_SIZE])
    img_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
    img_padded[:h, :w, :] = img.copy()  # 按照最长边进行padding，边长的最小值是INPUT_SIZE
    img = img_padded
    # 把图片resize到512的尺寸
    new_h, new_w, _ = img.shape
    img = cv2.resize(img, dsize=(INPUT_SIZE, INPUT_SIZE))
    resize_ratio_x = INPUT_SIZE / float(new_h)
    resize_ratio_y = INPUT_SIZE / float(new_w)
    text_polys[:, :, 0] *= resize_ratio_x
    text_polys[:, :, 1] *= resize_ratio_y
    new_h, new_w, _ = img.shape

    # 生成矩形框，这里的矩形是基于四个顶点坐标，带有旋转角度
    score_map, geo_map, training_mask, rectangles = generate_rbox((new_h, new_w), text_polys, text_tags)

    text_label = [text_label[i] for i in selected_poly]
    mask = [not (word == [-1]) for word in text_label]
    text_label = list(compress(text_label, mask))
    rectangles = list(compress(rectangles, mask))
    # return img,  # text_polys, text_tags, text_label
    # return img, score_map, geo_map, training_mask, transform_matrixes, boxes_masks, box_widths, text_labels_sparse


if __name__ == '__main__':
    # ------------first method-----
    # df = pd.read_csv('./icdar/train/train.csv')
    # file_paths = df['file_name'].values
    # gt_paths = df['gt'].values
    # ds_train = tf.data.Dataset.from_tensor_slices((file_paths, gt_paths))
    # ds_train = ds_train.map(lambda file_path, gt_path: tf.numpy_function(
    #     func=read_img,
    #     inp=(file_path, gt_path),
    #     Tout=tf.uint8,
    # ))
    #
    # # 验证是否正确
    # for i, info in enumerate(ds_train):
    #     print(info.shape)
    #     plt.imshow(info)
    #     plt.show()
    #     print('ok')
    #     break

    # ----------second method---------------
    img_list = open('./icdar/train/images.txt', 'r').readlines()
    img_list = [img_mem.strip() for img_mem in img_list]
    gt_list = open('./icdar/train/GT.txt', 'r').readlines()
    gt_list = [gt_mem.strip() for gt_mem in gt_list]

    # ----通过tf.data.Dataset.from_generator产生输入数据
    # Generator = generator(img_list, gt_list)
    # dataset = tf.data.Dataset.from_generator(
    #     generator,
    #     tf.int32,
    # ).batch(3)

    # ---测试generator的正确性
    images, image_fns, score_maps, geo_maps, training_masks, transform_matrixes, boxes_masks, box_widths, text_labels_sparse = next(generator(
        img_list, gt_list))
    # 显示图像
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.title('image')
    plt.imshow(images[0][::4, ::4,:]/255.)
    plt.subplot(2, 2, 2)
    plt.title('score_maps')
    plt.set_cmap('binary')
    plt.imshow(score_maps[0])
    plt.subplot(2, 2, 3)
    plt.title('training_mask')
    plt.set_cmap('binary')
    plt.imshow(training_masks[0])
    plt.show()
    # 显示真实的标签信息
    number = 0
    text_info = ''
    for i in range(len(text_labels_sparse[1])):
        str_index = text_labels_sparse[1][i]
        text_info += (config.CHAR_VECTOR[str_index])
    print('------文字部分的标签信息可视化----------')
    print(f'每一个字符对应的索引: {text_labels_sparse[1]}')
    print(f'对应的索引转化为字符: {text_info}')
