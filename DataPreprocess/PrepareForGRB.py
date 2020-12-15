import numpy as np
from shapely.geometry import Polygon


def shrink_poly(poly, r):
    '''
    fit a poly inside the origin poly, maybe bugs here...
    used for generate the score map
    :param poly: the text poly
    :param r: r in the paper
    :return: the shrinked poly
    '''
    # shrink ratio 用以调整缩小的范围，也就是在矩形框中找出一个范围稍小的矩形框，缩小的参数为R
    R = 0.3
    # find the longer pair
    if np.linalg.norm(poly[0] - poly[1]) + np.linalg.norm(poly[2] - poly[3]) > \
            np.linalg.norm(poly[0] - poly[3]) + np.linalg.norm(poly[1] - poly[2]):
        # first move (p0, p1), (p2, p3), then (p0, p3), (p1, p2)
        ## p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
        ## p0, p3
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
    else:
        ## p0, p3
        # print poly
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
        ## p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
    return poly


def fit_line(p1, p2):
    '''
    两点确定一条直线
    :param p1: 两个点的横坐标
    :param p2: 两个点的纵坐标
    :return: 确定的直线 kx-y+b=0的格式
    '''
    if p1[0] == p1[1]:
        return [1., 0., -p1[0]]
    else:
        [k, b] = np.polyfit(p1, p2, deg=1)
        return [k, -1, b]


def point_dist_to_line(p1, p2, p3):
    '''
    点到直线的距离公式
    :param p1: 直线上一点
    :param p2: 直线上一点
    :param p3: 直线外一点，
    :return: p3到直线p1-p2的距离
    '''
    return np.linalg.norm(np.cross(p2 - p3, p1 - p3)) / np.linalg.norm(p2 - p1)


def line_cross_point(line1, line2):
    '''
    计算两个直线的交点
    :param line1: kx-y+b=0
    :param line2:
    :return:
    '''
    if line1[0] != 0 and line1[0] == line2[0]:
        print('交点不存在')
        return None

    if line1[0] == 0 and line2[0] == 0:
        print('交点不存在')
        return None

    if line1[1] == 0:  # 这条直线是平行于y轴
        x = -line1[2]
        y = line2[0] * x + line2[2]
    elif line2[1] == 0:
        x = -line2[2]
        y = line1[0] * x + line1[2]
    else:
        k1, _, b1 = line1
        k2, _, b2 = line2
        x = -(b1 - b2) / (k1 - k2)
        y = k1 * x + b1
    return np.array([x, y], dtype=np.float32)


def line_verticle(line, point):
    '''
    求过直线外点point的直线line的法线
    :param line:
    :param point:
    :return:
    '''
    # 如果line是平行于y轴，法线就是平行于x轴
    if line[1] == 0:
        verticle = [0, -1, point[1]]
    else:
        if line[0] == 0:  # line是平行于x轴
            verticle = [1, 0, -point[0]]
        else:
            verticle = [-1./line[0], -1, point[1]-(-1/line[0]*point[0])]
    return verticle


def rectange_from_parallelogram(poly):
    '''

    :param poly:
    :return:
    '''
    p0, p1, p2, p3 = poly
    # 求p0处的的角度即线段p0p1, p0p3的夹角
    angle_p0 = np.arccos(np.dot(p1-p0, p3-p0) / (np.linalg.norm(p0-p1)*(np.linalg.norm(p3-p0))))
    if angle_p0 < 0.5*np.pi:  # 夹角小于90度
        if np.linalg.norm(p0-p1) > np.linalg.norm(p0-p3):  # 如果p0p1的边较长
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])  # 求p2p3这条直线
            p2p3_verticle = line_verticle(p2p3, p0)  # 求p2p3上的过p0的法线
            new_p3 = line_cross_point(p2p3, p2p3_verticle)

            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p2)
            new_p1 = line_cross_point(p0p1, p0p1_verticle)

            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
        else:
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_verticle = line_verticle(p1p2, p0)
            new_p1 = line_cross_point(p1p2, p1p2_verticle)

            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p2)
            new_p3 = line_cross_point(p0p3, p0p3_verticle)

            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
    else:
        if np.linalg.norm(p0 - p1) > np.linalg.norm(p0 - p3):  # 如果p0p1的边较长
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])  # 求p2p3这条直线
            p2p3_verticle = line_verticle(p2p3, p1)  # 求p2p3上的过p0的法线
            new_p2 = line_cross_point(p2p3, p2p3_verticle)

            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p3)
            new_p0 = line_cross_point(p0p1, p0p1_verticle)

            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)
        else:
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_verticle = line_verticle(p1p2, p3)
            new_p2 = line_cross_point(p1p2, p1p2_verticle)

            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p1)
            new_p0 = line_cross_point(p0p3, p0p3_verticle)

            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)


def sort_rectangle(poly):
    '''
    对于一个任意角度的矩形，找到它的旋转角度,旋转角度的范围是 [-45, 45]之间
    :param poly:
    :return:
    '''
    p_lowest = np.argmax(poly[:, 1]) # 找到四个顶点中的最低点，也就是y坐标最大的点的index

    if np.count_nonzero(poly[:, 1] == poly[p_lowest, 1]) == 2:  # y最大的点有两个，也就是矩形框的bottom是平行于x轴
        # p0就是横纵坐标之和最小的点，因为poly此时是矩形，且四条边分别平行于坐标轴,四个顶点是顺时针排序
        p0_index = np.argmin(np.sum(poly, axis=1))
        p1_index = (p0_index + 1) % 4
        p2_index = (p0_index + 2) % 4
        p3_index = (p0_index + 3) % 4
        return poly[[p0_index, p1_index, p2_index, p3_index]], 0.
    else:
        p_lowest_right = (p_lowest - 1) % 4
        angle = np.arctan(
            -(poly[p_lowest][1] - poly[p_lowest_right][1]) / (poly[p_lowest][0] - poly[p_lowest_right][0])
        )
        if angle <= 0:
            print(angle, poly[p_lowest], poly[p_lowest_right])
        if angle / np.pi * 180 > 45:
            # 最低点是p2
            p2_index = p_lowest
            p1_index = (p2_index - 1) % 4
            p0_index = (p2_index - 2) % 4
            p3_index = (p2_index + 1) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], -(np.pi / 2 - angle)
        else:
            # 最低点是p3
            p3_index = p_lowest
            p0_index = (p3_index + 1) % 4
            p1_index = (p3_index + 2) % 4
            p2_index = (p3_index + 3) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], angle


def earn_rect_angle(poly):
    fitted_parallelograms = [] # 找出来的是四边形，不一定是矩形
    for i in range(4):
        p0 = poly[i]
        p1 = poly[(i + 1) % 4]
        p2 = poly[(i + 2) % 4]
        p3 = poly[(i + 3) % 4]
        edge = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
        backward_edge = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
        forward_edge = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
        if point_dist_to_line(p0, p1, p2) > point_dist_to_line(p0, p1, p3):
            if edge[1] == 0:  # 直线平行于y轴的情况
                edge_opposite = [1, 0, -p2[0]]
            else:  # 一般情况
                edge_opposite = [edge[0], -1, p2[1] - edge[0] * p2[0]]
        else:
            if edge[1] == 0:
                edge_opposite = [1, 0, -p3[0]]
            else:
                edge_opposite = [edge[0], -1, p3[1] - edge[0] * p3[0]]

        # new_p0 = p0
        # new_p1 = p1
        # new_p2 = p2
        # new_p3 = p3
        # move forward edge
        new_p2 = line_cross_point(forward_edge, edge_opposite)
        if point_dist_to_line(p1, new_p2, p0) > point_dist_to_line(p1, new_p2, p3):
            if forward_edge[1] == 0:
                forward_opposite = [1, 0, -p0[0]]
            else:
                forward_opposite = [forward_edge[0], -1, p0[1] - forward_edge[0] * p0[0]]
        else:
            if forward_edge[1] == 0:
                forward_opposite = [1, 0, -p3[0]]
            else:
                forward_opposite = [forward_edge[0], -1, p3[1] - forward_edge[0] * p3[0]]
        new_p0 = line_cross_point(forward_opposite, edge)
        new_p3 = line_cross_point(forward_opposite, edge_opposite)

        fitted_parallelograms.append([new_p0, p1, new_p2, new_p3, new_p0])

        # or move backward edge
        new_p3 = line_cross_point(backward_edge, edge_opposite)
        if point_dist_to_line(p0, p3, p1) > point_dist_to_line(p0, p3, p2):
            if backward_edge[1] == 0:
                backward_opposite = [1, 0, -p1[0]]
            else:
                backward_opposite = [backward_edge[0], -1, p1[1] - backward_edge[0] * p1[0]]
        else:
            if backward_edge[1] == 0:
                backward_opposite = [1, 0, -p2[0]]
            else:
                backward_opposite = [backward_edge[0], -1, p2[1] - backward_edge[0] * p2[0]]
        new_p1 = line_cross_point(backward_opposite, edge)
        new_p2 = line_cross_point(backward_opposite, edge_opposite)
        fitted_parallelograms.append([p0, new_p1, new_p2, new_p3, p0])
    areas = [Polygon(t).area for t in fitted_parallelograms]  # 计算多边形的面积

    # 排序 获取面积最小的四边框的四个坐标点
    parallelogram = np.array(fitted_parallelograms[np.argmin(areas)][:-1], dtype=np.float32)
    # 找到这四个坐标点中坐标和最小的坐标点，一般情况下是左上角坐标但是并不确定
    parallelogram_cood_sum = np.sum(parallelogram, axis=1)
    min_coord_idx = np.argmin(parallelogram_cood_sum)
    parallelogram = parallelogram[[
        min_coord_idx,
        (min_coord_idx + 1) % 4,
        (min_coord_idx + 2) % 4,
        (min_coord_idx + 3) % 4
    ]]
    rectange = rectange_from_parallelogram(parallelogram) # 从找出来的四边形中产生对应的矩形框
    rectange, rotate_angle = sort_rectangle(rectange)
    return rectange, rotate_angle