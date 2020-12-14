'''
画出任意四个顶点的最小外接矩形
'''
import numpy as np


# def make_rect(poly):
#     r = [None, None, None, None]
#     for i in range(4):
#         print(i)
#         r[i] = min(np.linalg.norm(poly[i] - poly[(i+1)%4]),
#                    np.linalg.norm(poly[i] - poly[(i-1)%4]))
#     print(f'r: {r}')
#
#     # 选择距离较长的两个临边，有两种情况
#     # 第一种
#     if (np.linalg.norm(poly[0]-poly[1])+np.linalg.norm(poly[2]-poly[3]) > \
#             np.linalg.norm(poly[0]-poly[3])+np.linalg.norm(poly[1]-poly[2])):
#         theta = np.arctan2((poly[1][1]-poly[0][1]), (poly[0][1]-poly[0][0])) # 获得两点的正切值，y/x


# coding:utf-8
import sys
# import codecs
# sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

import glob
import csv
import cv2
import time
import os
import math
import numpy as np
import scipy.optimize
from PIL import Image
# import matplotlib.pyplot as plt
# import pylab
# import matplotlib.patches as Patches
from itertools import compress
from shapely.geometry import Polygon
import random
import tensorflow as tf
import Aug_Operations as aug
import config


def shrink_poly(poly, r):
    '''
    fit a poly inside the origin poly, maybe bugs here...
    used for generate the score map
    :param poly: the text poly
    :param r: r in the paper
    :return: the shrinked poly
    '''
    # shrink ratio
    R = 0.3
    # find the longer pair
    if np.linalg.norm(poly[0] - poly[1]) + np.linalg.norm(poly[2] - poly[3]) > \
            np.linalg.norm(poly[0] - poly[3]) + np.linalg.norm(poly[1] - poly[2]):
        # first move (p0, p1), (p2, p3), then (p0, p3), (p1, p2)
        ## p0, p1
        print('ok------------------------------')
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


def point_dist_to_line(p1, p2, p3):
    # compute the distance from p3 to p1-p2
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)


def fit_line(p1, p2):
    # fit a line ax+by+c = 0
    if p1[0] == p1[1]:
        return [1., 0., -p1[0]]
    else:
        [k, b] = np.polyfit(p1, p2, deg=1)
        return [k, -1., b]


def line_cross_point(line1, line2):
    # line1 0= ax+by+c, compute the cross point of line1 and line2
    if line1[0] != 0 and line1[0] == line2[0]:
        print('Cross point does not exist')
        return None
    if line1[0] == 0 and line2[0] == 0:
        print('Cross point does not exist')
        return None
    if line1[1] == 0:
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
    # get the verticle line from line across point
    if line[1] == 0:
        verticle = [0, -1, point[1]]
    else:
        if line[0] == 0:
            verticle = [1, 0, -point[0]]
        else:
            verticle = [-1. / line[0], -1, point[1] - (-1 / line[0] * point[0])]
    return verticle


def rectangle_from_parallelogram(poly):
    '''
    fit a rectangle from a parallelogram
    :param poly:
    :return:
    '''
    p0, p1, p2, p3 = poly
    angle_p0 = np.arccos(np.dot(p1 - p0, p3 - p0) / (np.linalg.norm(p0 - p1) * np.linalg.norm(p3 - p0)))
    if angle_p0 < 0.5 * np.pi:
        if np.linalg.norm(p0 - p1) > np.linalg.norm(p0 - p3):
            # p0 and p2
            ## p0
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_verticle = line_verticle(p2p3, p0)

            new_p3 = line_cross_point(p2p3, p2p3_verticle)
            ## p2
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
        if np.linalg.norm(p0 - p1) > np.linalg.norm(p0 - p3):
            # p1 and p3
            ## p1
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_verticle = line_verticle(p2p3, p1)

            new_p2 = line_cross_point(p2p3, p2p3_verticle)
            ## p3
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p3)

            new_p0 = line_cross_point(p0p1, p0p1_verticle)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)
        else:
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p1)

            new_p0 = line_cross_point(p0p3, p0p3_verticle)
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_verticle = line_verticle(p1p2, p3)

            new_p2 = line_cross_point(p1p2, p1p2_verticle)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)


def distance(poly):
    '''
    :计算两个顶点的距离，用于计算矩形框的边长
    :param poly:
    :return:
    '''
    point1 = poly[[0]]
    point2 = poly[[1]]
    dis = np.sqrt(np.sum((point1 - point2) * (point1 - point2)))
    return dis


def sort_rectangle(poly):
    # sort the four coordinates of the polygon, points in poly should be sorted clockwise
    # First find the lowest point
    p_lowest = np.argmax(poly[:, 1])

    if np.count_nonzero(poly[:, 1] == poly[p_lowest, 1]) == 2:
        # 底边平行于X轴, 那么p0为左上角 - if the bottom line is parallel to x-axis, then p0 must be the upper-left corner
        p0_index = np.argmin(np.sum(poly, axis=1))
        p1_index = (p0_index + 1) % 4
        p2_index = (p0_index + 2) % 4
        p3_index = (p0_index + 3) % 4
        return poly[[p0_index, p1_index, p2_index, p3_index]], 0.
    else:
        # 找到最低点右边的点 - find the point that sits right to the lowest point
        p_lowest_right = (p_lowest - 1) % 4
        p_lowest_left = (p_lowest + 1) % 4
        angle = np.arctan(
            -(poly[p_lowest][1] - poly[p_lowest_right][1]) / (poly[p_lowest][0] - poly[p_lowest_right][0]))
        # assert angle > 0
        if angle <= 0:
            print(angle, poly[p_lowest], poly[p_lowest_right])
        if angle / np.pi * 180 > 45:
            # 这个点为p2 - this point is p2
            p2_index = p_lowest
            p1_index = (p2_index - 1) % 4
            p0_index = (p2_index - 2) % 4
            p3_index = (p2_index + 1) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], -(np.pi / 2 - angle)
        else:
            # 这个点为p3 - this point is p3
            p3_index = p_lowest
            p0_index = (p3_index + 1) % 4
            p1_index = (p3_index + 2) % 4
            p2_index = (p3_index + 3) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], angle


def generate_rbox(im_size, poly, tag):
    h, w = im_size
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    score_map = np.zeros((h, w), dtype=np.uint8)
    geo_map = np.zeros((h, w, 5), dtype=np.float32)
    # mask used during traning, to ignore some hard areas
    training_mask = np.ones((h, w), dtype=np.uint8)
    rectangles = []

    r = [None, None, None, None]
    for i in range(4):
        r[i] = min(np.linalg.norm(poly[i] - poly[(i + 1) % 4]),
                   np.linalg.norm(poly[i] - poly[(i - 1) % 4]))

    shrinked_poly = shrink_poly(poly.copy(), r).astype(np.int32)[np.newaxis, :, :]
    print(shrinked_poly)
    print(r)

    poly_idx = 0

    cv2.fillPoly(score_map, shrinked_poly, 1)
    cv2.fillPoly(poly_mask, shrinked_poly, poly_idx + 1)
    # if the poly is too small, then ignore it during training
    poly_h = min(np.linalg.norm(poly[0] - poly[3]), np.linalg.norm(poly[1] - poly[2]))
    poly_w = min(np.linalg.norm(poly[0] - poly[1]), np.linalg.norm(poly[2] - poly[3]))
    if min(poly_h, poly_w) < 400:#FLAGS.min_text_size:
        cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
    if tag:
        cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)

    xy_in_poly = np.argwhere(poly_mask == (poly_idx + 1))
    # if geometry == 'RBOX':
    # 对任意两个顶点的组合生成一个平行四边形 - generate a parallelogram for any combination of two vertices
    fitted_parallelograms = []
    for i in range(4):
        p0 = poly[i]
        p1 = poly[(i + 1) % 4]
        p2 = poly[(i + 2) % 4]
        p3 = poly[(i + 3) % 4]
        edge = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
        backward_edge = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
        forward_edge = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
        if point_dist_to_line(p0, p1, p2) > point_dist_to_line(p0, p1, p3):
            # 平行线经过p2 - parallel lines through p2
            if edge[1] == 0:
                edge_opposite = [1, 0, -p2[0]]
            else:
                edge_opposite = [edge[0], -1, p2[1] - edge[0] * p2[0]]
        else:
            # 经过p3 - after p3
            if edge[1] == 0:
                edge_opposite = [1, 0, -p3[0]]
            else:
                edge_opposite = [edge[0], -1, p3[1] - edge[0] * p3[0]]
        # move forward edge
        new_p0 = p0
        new_p1 = p1
        new_p2 = p2
        new_p3 = p3
        new_p2 = line_cross_point(forward_edge, edge_opposite)
        if point_dist_to_line(p1, new_p2, p0) > point_dist_to_line(p1, new_p2, p3):
            # across p0
            if forward_edge[1] == 0:
                forward_opposite = [1, 0, -p0[0]]
            else:
                forward_opposite = [forward_edge[0], -1, p0[1] - forward_edge[0] * p0[0]]
        else:
            # across p3
            if forward_edge[1] == 0:
                forward_opposite = [1, 0, -p3[0]]
            else:
                forward_opposite = [forward_edge[0], -1, p3[1] - forward_edge[0] * p3[0]]
        new_p0 = line_cross_point(forward_opposite, edge)
        new_p3 = line_cross_point(forward_opposite, edge_opposite)
        fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
        # or move backward edge
        new_p0 = p0
        new_p1 = p1
        new_p2 = p2
        new_p3 = p3
        new_p3 = line_cross_point(backward_edge, edge_opposite)
        if point_dist_to_line(p0, p3, p1) > point_dist_to_line(p0, p3, p2):
            # across p1
            if backward_edge[1] == 0:
                backward_opposite = [1, 0, -p1[0]]
            else:
                backward_opposite = [backward_edge[0], -1, p1[1] - backward_edge[0] * p1[0]]
        else:
            # across p2
            if backward_edge[1] == 0:
                backward_opposite = [1, 0, -p2[0]]
            else:
                backward_opposite = [backward_edge[0], -1, p2[1] - backward_edge[0] * p2[0]]
        new_p1 = line_cross_point(backward_opposite, edge)
        new_p2 = line_cross_point(backward_opposite, edge_opposite)
        fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
    areas = [Polygon(t).area for t in fitted_parallelograms]
    parallelogram = np.array(fitted_parallelograms[np.argmin(areas)][:-1], dtype=np.float32)
    # sort thie polygon
    parallelogram_coord_sum = np.sum(parallelogram, axis=1)
    min_coord_idx = np.argmin(parallelogram_coord_sum)
    parallelogram = parallelogram[
        [min_coord_idx, (min_coord_idx + 1) % 4, (min_coord_idx + 2) % 4, (min_coord_idx + 3) % 4]]

    rectange = rectangle_from_parallelogram(parallelogram)
    rectange, rotate_angle = sort_rectangle(rectange)
    print(rectange, rotate_angle)
    # rectangles.append(rectange.flatten())
    # p0_rect, p1_rect, p2_rect, p3_rect = rectange

    '''
        # geo_thread阈值设置为矩形短边的两倍（认为短边为一个字大小）
        geo_thread = 0
        if np.linalg.norm(p0_rect - p1_rect) > np.linalg.norm(p1_rect - p2_rect):
            geo_thread = np.linalg.norm(p1_rect - p2_rect)
        else:
            geo_thread = np.linalg.norm(p0_rect - p1_rect)
        # 2表示点到边的距离按照两个字来算
        geo_thread = 2 * geo_thread
        # print("geo_thread:{}".format(geo_thread))
        assert geo_thread > 0

        for y, x in xy_in_poly:
            point = np.array([x, y], dtype=np.float32)
            # top
            geo_map[y, x, 0] = min(point_dist_to_line(p0_rect, p1_rect, point), geo_thread)
            # right
            geo_map[y, x, 1] = min(point_dist_to_line(p1_rect, p2_rect, point), geo_thread)
            # down
            geo_map[y, x, 2] = min(point_dist_to_line(p2_rect, p3_rect, point), geo_thread)
            # left
            geo_map[y, x, 3] = min(point_dist_to_line(p3_rect, p0_rect, point), geo_thread)
            # angle
            geo_map[y, x, 4] = rotate_angle
    return score_map, geo_map, training_mask, rectangles
    '''
    return 0


def distance_2p(point1, point2):
    # point1 = poly[[0]]
    # point2 = poly[[1]]
    dis = np.sqrt(np.sum((point1 - point2) * (point1 - point2)))
    return dis


def get_project_matrix_and_width(text_polyses, target_height=8.0):
    project_matrixes = []
    box_widths = []
    filter_box_masks = []
    #    max_width = 256
    # max_width = 0

    for i in range(text_polyses.shape[0]):
        x1, y1, x2, y2, x3, y3, x4, y4 = text_polyses[i] / 2
        width = distance_2p(np.array([x1, y1]), np.array([x2, y2]))
        height = distance_2p(np.array([x1, y1]), np.array([x4, y4]))
        # width_box = int(min(width_box, 512)) # not to exceed feature map's width

        # 斜体增强
        ratio = random.uniform(0, 1)
        if ratio < 0.3 and width > 2 * height:
            delta = 0.5 * height
            x1 = x1 - delta
            x3 = x3 + delta
        elif ratio < 0.6 and width > 2 * height:
            delta = 0.5 * height
            x2 = x2 + delta
            x4 = x4 - delta
        # mapped_x3, mapped_y3 = (width_box, 8)
        delta1 = random.uniform(-0.1, 0.1)
        delta2 = random.uniform(-0.1, 0.1)

        delta3 = random.uniform(-0.3, 0.3)
        delta4 = random.uniform(-0.3, 0.3)
        # print delta
        if width > 1.5 * height:
            width_box = math.ceil(8 * width / height)
            src_pts = np.float32(
                [(x1 + delta3 * height, y1 + delta1 * height), (x2 + delta4 * height, y2 + delta1 * height),
                 (x4 + delta3 * height, y4 + delta2 * height)])
        else:
            width_box = math.ceil(8 * height / width)
            src_pts = np.float32(
                [(x2 + delta2 * width, y2 + delta3 * width), (x3 + delta2 * width, y3 + delta4 * width),
                 (x1 + delta1 * width, y1 + delta3 * width)])
        mapped_x1, mapped_y1 = (0, 0)
        mapped_x4, mapped_y4 = (0, 8)
        mapped_x2, mapped_y2 = (width_box, 0)
        dst_pts = np.float32([(mapped_x1, mapped_y1), (mapped_x2, mapped_y2), (mapped_x4, mapped_y4)])

        affine_matrix = cv2.getAffineTransform(dst_pts.astype(np.float32), src_pts.astype(np.float32))
        affine_matrix = affine_matrix.flatten()

        project_matrixes.append(affine_matrix)
        box_widths.append(width_box)

    project_matrixes = np.array(project_matrixes)
    box_widths = np.array(box_widths)

    return project_matrixes, box_widths


if __name__ == '__main__':
    # 四个点坐标
    # p = [[602, 173], [634, 173], [634, 196], [602, 196]]
    # p = [[602, 173], [634, 200], [634, 300], [602, 400]]
    p = [[2, 1], [6, 2], [6, 3], [3, 3]]
    point = np.array(p, dtype=np.int)
    img_size = (720, 1280)
    generate_rbox(img_size, point, 0)
    # make_rect(point)