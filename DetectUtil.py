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
        '''
        解析：
            origin_0的坐标是原始图片中的坐标，假设我们要检测的图片是224*224，则origin_0中的坐标与真实图片坐标是对应的
            d_0中的距离是变换后的距离，经过rotate后产生的即为检测图片中的距离
            p中相当于存放矩形框，
            origin_0中的坐标与矩形框各个边的距离在d_0中，因此，由origin_0的坐标值和d_0的距离可推算矩形框在原图中的位置
            关于旋转变换中的正负号不太明白？？
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
        print(f'd_0: {d_0}')
        print(f'p: {p}')
        print(f'angle_0: {angle_0}')
        rotate_matrix_x = np.array([np.cos(angle_0), np.sin(angle_0)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        rotate_matrix_y = np.array([-np.sin(angle_0), np.cos(angle_0)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x*p, axis=2)[:, :, np.newaxis]
        p_rotate_y = np.sum(rotate_matrix_y*p, axis=2)[:, :, np.newaxis]
        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)
        p3_in_origin = origin_0 - p_rotate[:, 4, :]
        print(f'p_rotate: {p_rotate}')
        print(f'origin_0: {origin_0}')
        print(f'p3_in_origin: {p3_in_origin}')
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_0 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
        print(f'new_p_0: {new_p_0}')

    # 对于角度小于0的情况
    origin_1 = origin[angle < 0]
    d_1 = d[angle < 0]
    angle_1 = angle[angle < 0]
    if origin_1.shape[0] > 0:
        pass
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

        text_box_restored, angle_m = restore_rectangle(xy_text[:, ::-1]*4,
                                                       geo_map[xy_text[:, 0], xy_text[:, 1], :])

    return 0


if __name__ == '__main__':
    score = np.load('./score.npy')
    geometry = np.load('./geometry.npy')  # 第四维存放的是角度
    share_data = np.load('./share_data.npy')
    print(score.shape, geometry.shape, share_data.shape)
    detect_contours(score, geometry)


    x= [[20.1339746, 23.59807621], [23.59807621, 20.59807621], [20.59807621, 17.1339746], [17.1339746,20.1339746]]
    y=[[17.76794919,19.76794919],[19.76794919,24.96410162],[24.96410162,22.96410162],[22.96410162,17.76794919]]


