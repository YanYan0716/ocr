from DataPreprocess.DataGen import load_annoatation
import numpy as np


if __name__ == '__main__':
    gt_list = open('./icdar/train/GT.txt', 'r').readlines()
    gt_list = [gt_mem.strip() for gt_mem in gt_list]
    gt_list = np.array(gt_list)

    for i in range(len(gt_list)):
        text_polys, text_tags, text_label = load_annoatation(gt_list[i])
        # print(text_label)
    # label = label.replace(' ', '')
    # label = label.replace('')
    # return [config.CHAR_VECTOR.index(x) for x in label]
    print(i)