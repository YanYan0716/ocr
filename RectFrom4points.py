'''
画出任意四个顶点的最小外接矩形
'''
import numpy as np


def make_rect(poly):
    r = [None, None, None, None]
    for i in range(4):
        print(i)
        r[i] = min(np.linalg.norm(poly[i] - poly[(i+1)%4]),
                   np.linalg.norm(poly[i] - poly[(i-1)%4]))
    print(f'r: {r}')

    # 选择距离较长的两个临边，有两种情况
    # 第一种
    if (np.linalg.norm(poly[0]-poly[1])+np.linalg.norm(poly[2]-poly[3]) > \
            np.linalg.norm(poly[0]-poly[3])+np.linalg.norm(poly[1]-poly[2])):
        print('ok')



if __name__ == '__main__':
    # 四个点坐标
    p = [[602, 173], [634, 173], [634, 196], [602,196]]
    point = np.array(p, dtype=np.int)
    make_rect(point)