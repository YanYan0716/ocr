import cv2


def ResizeImg(img, max_h=1000, max_w=1000):
    '''
    限制图像的最大尺寸，同时要保持原比例
    :param img:
    :param max_h:
    :param max_w:
    :return:
    '''
    h, w, chn = img.shape
    ratio = 1.0
    max_flag = max(h, w)
    if max_flag > 1000:
        ratio = max_flag / max_h
    resize_h = h / ratio
    resize_w = w / ratio
    img = cv2.resize(img, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return img, (ratio_h, ratio_w)
