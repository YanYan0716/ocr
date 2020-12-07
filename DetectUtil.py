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
