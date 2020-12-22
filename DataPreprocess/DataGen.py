import tensorflow as tf
import cv2
import numpy as np

from DataPreprocess.DataGenerator import load_annoatation, check_and_validate_polys, img_aug, crop_area


class DataGen:
    def __init__(self, image_list, gt_list, batch_size=4, ):
        self.i = 0
        self.image_list = image_list
        self.gt_list = gt_list
        self.batch_size = batch_size
        self.random_scale = np.array([0.5, 0.6, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.4, 1.6, 2, 3, 4]

    def __load__(self, files_name, txt_fn):
        '''
        读取一个image和gt的信息
        :param files_name: image的文件名
        :param txt_fn: 相关gt的文件名
        :return:
        '''
        img = cv2.imread(files_name)
        if img is None:
            self.__next__()
        h, w, _ = img.shape

        # 返回的数据类型：np.array, np.array, list
        text_polys, text_tags, text_label = load_annoatation(txt_fn)

        # 检查获取到的矩形框四个顶点的顺序对不对，过滤掉面积为0的情况
        text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, h, w)
        # print(text_polys.shape, text_tags)
        rd_scale = np.random.choice(self.random_scale)  # 随机选一个scale

        img = cv2.resize(img, dsize=None, fx=rd_scale, fy=rd_scale)
        file_name = files_name.split('/')[-1]
        img = img_aug(img, file_name)
        text_polys *= rd_scale

        # 对图片进行裁剪
        img, text_polys, text_tags, selected_poly = crop_area(img, text_polys, text_tags, crop_background=False)
        # print(text_polys.shape, text_tags)
        if text_polys.shape[0] == 0 or len(text_label) == 0:
            self.__next__()
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














        print(files_name)
        img = cv2.imread(files_name)
        return img

    def getitem(self, index):
        _img = self.__load__(self.image_list[index], self.gt_list[index])
        return _img

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < self.batch_size:
            img_arr = self.getitem(self.i)
            self.i += 1
        else:
            raise StopIteration()
        return img_arr

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
        (tf.int32),
    )

    # for j in range(2):
    #     print('---------------------')
    #     for i in dataset:
    #         print(i.shape)

    dataset = dataset.batch(2).shuffle()
    for i in dataset:
        print(i.shape)