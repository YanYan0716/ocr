import numpy as np

if __name__ == '__main__':
    # x= [[20.1339746, 23.59807621], [23.59807621, 20.59807621], [20.59807621, 17.1339746], [17.1339746,20.1339746]]
    # y=[[17.76794919,19.76794919],[19.76794919,24.96410162],[24.96410162,22.96410162],[22.96410162,17.76794919]]
    points = np.array([[0, 0], [6, 0], [6, -4], [0, -4], [4, -3]])

    angle = [60 / 180 * np.pi]
    # # 对于角度大于0 顺时针
    # # x' = x*cos+y*sin  y'=-x*sin+ycos
    # rotate_matrix_x = np.array([np.cos(angle), np.sin(angle)]).transpose((1, 0))
    # rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(2, 5).transpose((1, 0))
    #
    # rotate_matrix_y = np.array([-np.sin(angle), np.cos(angle)]).transpose((1, 0))
    # rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(2, 5).transpose((1, 0))
    #
    # new_pointX = np.sum(points * rotate_matrix_x, axis=1)[:, np.newaxis]
    # new_pointY = np.sum(points * rotate_matrix_y, axis=1)[:, np.newaxis]
    # new_point = np.concatenate([new_pointX, new_pointY], axis=1)
    #
    # new_point = new_point
    #
    # pp = [20, 20] - new_point[4]
    # new_point = new_point + pp

    # 对应于角度小于0
