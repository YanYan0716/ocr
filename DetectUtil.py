import numpy as np
from PIL import Image
import cv2


def detect_contours(score_map, geo_map, timer, score_map_thresh=0.8, boc_thresh=0.1, nms_thres=0.1):
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
    score_img = score_map*255
    kernel = np.uint8(np.ones((1, 3)))
    score_img = cv2.dilate(score_img, kernel) # 膨胀处理
    im = cv2.erode(score_img, kernel)
    im = Image.fromarray(im)
    return 0
