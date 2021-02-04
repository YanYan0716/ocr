import numpy as np
from PIL import Image
import cv2
# import lanms
import matplotlib.pyplot as plt


import locality_aware_nms as nms_locality


def restore_rectangle(origin, geometry):
    '''
    每调用一次这个函数都是对某个矩形框中的所有像素点所带油的信息做操作，所以返回np.mean，相当于是这个矩形框中所有像素点角度
    的均值，代表着矩形框的均值
    :param origin: shaoe:[N, 2] 其中的每一行都是一个像素的坐标
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
        '''
        解析：
            origin_0的坐标是原始图片中的坐标，假设我们要检测的图片是224*224，则origin_0中的坐标与真实图片坐标是对应的
            d_0中的距离是变换后的距离，经过rotate后产生的即为检测图片中的距离
            p中相当于存放矩形框，
            origin_0中的坐标与矩形框各个边的距离在d_0中，因此，由origin_0的坐标值和d_0的距离可推算矩形框在原图中的位置
            关于旋转变换中的正负号不太明白？？
            在二维坐标转换    顺时针:  x'=x*cos(a)+y*sin(a)    y'=-x*sin(a)+y*cos(a)
                            逆时针:  x'=x*cos(a)-y*sin(a)    y'= x*sin(a)+y*cos(a)
                            这其中的角度均为正数
            更具体理解请见rotateRect.py
        '''
        neg_width = -d_0[:, 0] - d_0[:, 2] # 宽的相反数
        length = d_0[:, 1]+d_0[:, 3] # 长
        # p shape: [10, N]
        p = np.array([np.zeros(d_0.shape[0]),
                      neg_width,
                      length,
                      neg_width,
                      length,
                      np.zeros(d_0.shape[0]),
                      np.zeros(d_0.shape[0]),
                      np.zeros(d_0.shape[0]),
                      d_0[:, 3],
                      -d_0[:, 2]])
        p = p.transpose((1, 0)).reshape((-1, 5, 2)) # p shape:[N, 5, 2]
        # print(f'd_0: {d_0}')
        # print(f'p: {p}')
        # print(f'angle_0: {angle_0}')
        rotate_matrix_x = np.array([np.cos(angle_0), np.sin(angle_0)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        rotate_matrix_y = np.array([-np.sin(angle_0), np.cos(angle_0)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x*p, axis=2)[:, :, np.newaxis]
        p_rotate_y = np.sum(rotate_matrix_y*p, axis=2)[:, :, np.newaxis]
        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)
        p3_in_origin = origin_0 - p_rotate[:, 4, :]
        # print(f'p_rotate: {p_rotate}')
        # print(f'origin_0: {origin_0}')
        # print(f'p3_in_origin: {p3_in_origin}')
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_0 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
        # print(f'new_p_0: {new_p_0}')
    else:
        new_p_0 = np.zeros((0, 4, 2))

    # 对于角度小于0的情况
    origin_1 = origin[angle < 0]
    d_1 = d[angle < 0]
    angle_1 = angle[angle < 0]
    if origin_1.shape[0] > 0:
        neg_length = -d_1[:, 1]-d_1[:, 3]
        neg_width = -d_1[:, 0]-d_1[:, 2]
        p = np.array([
            neg_length,
            neg_width,
            np.zeros(d_1.shape[0]),
            neg_width,
            np.zeros(d_1.shape[0]),
            np.zeros(d_1.shape[0]),
            neg_length,
            np.zeros(d_1.shape[0]),
            -d_1[:, 1],
            -d_1[:, 2]
        ])
        # print(p)
        p = p.transpose((1, 0)).reshape((-1, 5, 2))
        # 使用angle_1的负号是因为本身为负数，而公式中的角度按顺时针于逆时针的都是正数
        rotate_matrix_x = np.array([np.cos(-angle_1), -np.sin(-angle_1)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        rotate_matrix_y = np.array([np.sin(-angle_1), np.cos(-angle_1)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x*p, axis=2)[:,:, np.newaxis]
        p_rotate_y = np.sum(rotate_matrix_y*p, axis=2)[:,:, np.newaxis]

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)

        p3_in_origin = origin_1 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_1 = np.concatenate([new_p0[:, np.newaxis, :],
                                 new_p1[:, np.newaxis, :],
                                 new_p2[:, np.newaxis, :],
                                 new_p3[:, np.newaxis, :]], axis=1)
    else:
        new_p_1 = np.zeros((0, 4, 2))
    return np.concatenate([new_p_0, new_p_1]), np.mean(angle)


def detect_contours(score_map, geo_map, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.1):
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

        text_box_restored, angle_m = restore_rectangle(xy_text[:, ::-1]*4,
                                                       geo_map[xy_text[:, 0], xy_text[:, 1], :])
        # print(text_box_restored.reshape((-1, 8)), angle_m)
        # 返回的是所有点的坐标集合,如果旋转角度较大，那么从这些点集中找到最小外接矩形
        # if angle_m/np.pi > 10 or angle_m/np.pi < -10:
        #     points = text_box_restored.reshape((-1, 2))
        #     rect = cv2.minAreaRect(points.astype(np.int32))
        #     rec_box = cv2.boxPoints(rect)
        #     score_sum = np.sum(score_map[xy_text[:, 0], xy_text[:, 1]])
        #     rec_box = np.append(rec_box.reshape(-1, 8), score_sum)
        # else:
        #     boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
        #     boxes[:, :8] = text_box_restored.reshape((-1, 8))
        #     boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
        #     print(boxes)
        #     # rec_box = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
        #     # rec_box = lanms.rec_standard_nms(boxes.astype('float32'), nms_thres)
        # res_boxes.append(rec_box)
        score_sum = np.sum(score_map[xy_text[:, 0], xy_text[:, 1]])
        rec_box = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
        rec_box[:, :8] = text_box_restored.reshape((-1, 8))
        rec_box[:, 8] = score_sum
        # break
        res_boxes.append(rec_box)
    boxes =np.squeeze(np.array(res_boxes), axis=0)
    # print(boxes)
    # print(boxes.shape)
    boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    # print(boxes)
    # # boxes = lanms.merge_quadrangle_n9(np.array(res_boxes).astype('int'), nms_thres)
    boxes_list = boxes.tolist()
    boxes_list = sorted(boxes_list, key=lambda k: [k[1], k[0]])
    boxes = np.array(boxes_list)

    if boxes.shape[0] == 0:
        return None

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
        # print(boxes[:, 8])
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis + 1) % 4, (min_axis + 2) % 4, (min_axis + 3) % 4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


if __name__ == '__main__':
    score = np.load('./score.npy')
    geometry = np.load('./geometry.npy')  # 第四维存放的是角度
    share_data = np.load('./share_data.npy')
    print(score.shape, geometry.shape, share_data.shape)
    boxes = detect_contours(score, geometry)
    # print(boxes)
    # if boxes is not None and boxes.shape[0] != 0:
    #     boxes_detect = boxes.copy()
    #     boxes_detect = boxes_detect[:, :8].reshape((-1, 4, 2))
    #     im = cv2.imread('./123.jpg')
    #     for i , box in enumerate(boxes_detect):
    #         box = sort_poly(box.astype(np.int32))
    #         if np.linalg.norm(box[0]-box[1])<5 or np.linalg.norm(box[3]-box[0])<5:
    #             continue
    #         print(box)
    #         cv2.polylines(im, [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
    #     cv2.imwrite('./result.jpg', im)