import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt


def restore_rectangle(origin, geometry):
    '''

    :param origin: shaoe:[N, 2]
    :param geometry: shape:[N, 5]
    :return:
    '''
    d = geometry[:, :4]
    angle = geometry[:, 4]
    # 对于角度大于0的情况
    origin_0 = origin[angle >= 0]
    d_0 = d[angle >= 0]
    angle_0 = angle[angle >= 0]
    if origin_0.shape[0] > 0:
        pass
    # 对于角度小于0的情况
    origin_1 = origin[angle < 0]
    d_1 = d[angle < 0]
    angle_1 = angle[angle < 0]
    if origin_1.shape[0] > 0:
        p = np.array([-d_1[:, 1] - d_1[:, 3], -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), np.zeros(d_1.shape[0]),
                      -d_1[:, 1] - d_1[:, 3], np.zeros(d_1.shape[0]),
                      -d_1[:, 1], -d_1[:, 2]])
        print(p.shape)
    return 0, 1,


def detect_contours(score_map, geo_map, score_map_thresh=0.8, boc_thresh=0.1, nms_thres=0.1):
    '''
    从网络预测中得到可理解的结果
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh:
    :param boc_thresh:
    :param nms_thres:
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]

    h, w = score_map.shape[:2]
    score_img = score_map * 255
    kernel = np.uint8(np.ones((1, 3)))
    score_img = cv2.dilate(score_img, kernel)  # 膨胀处理

    # # 显示score_map中的内容
    # im = cv2.erode(score_img, kernel)
    # im = Image.fromarray(im)
    # plt.imshow(im)
    # plt.show()

    im = np.array(score_img, np.uint8)
    # 图像二值化，大于阈值的设为255， 小于的设为0， 返回修改后的图像
    ret, im = cv2.threshold(im, score_map_thresh * 255, 255, cv2.THRESH_BINARY)
    contours0, hierarchy = cv2.findContours(im.copy(),
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
    res_boxes = []
    for cnt in contours0:
        vis = np.zeros((h, w), np.uint8)
        # 获取近似多边形
        contours = cv2.approxPolyDP(cnt, 3, True)
        # 填充凸多边形
        cv2.fillConvexPoly(vis, np.int32(contours), (255, 255, 255))
        # 获取vis中像素值=255的点的坐标
        xy_text = np.argwhere(vis == 255)
        # 此处获得的坐标相当于原始图片缩小4倍之后的，所以下面的要*4
        xy_text = xy_text[np.argsort(xy_text[:, 0])]

        text_box_restored, angle_m = restore_rectangle(xy_text[:, ::-1],
                                                       geo_map[xy_text[:, 0], xy_text[:, 1], :])
        break

    return 0


if __name__ == '__main__':
    score = np.load('./score.npy')
    geometry = np.load('./geometry.npy')  # 第四维存放的是角度
    share_data = np.load('./share_data.npy')
    print(score.shape, geometry.shape, share_data.shape)
    detect_contours(score, geometry)
