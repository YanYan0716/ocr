import tensorflow as tf


def dice_coefficient(true_cls, pred_cls, training_mask):
    '''
    最理想的情况是true和pred相等，此时loss应该为0 所以在loss=...的那一行有一个scale：2
    :param true_cls:
    :param pred_cls:
    :param training_mask:
    :return:
    '''
    eps = 1e-5  # 保证除法中分母不为0
    inter = tf.reduce_sum(true_cls * pred_cls * training_mask)
    union_1 = tf.reduce_sum(true_cls * training_mask)
    union_2 = tf.reduce_sum(pred_cls * training_mask)
    union = (union_1 + union_2) + eps
    loss = 1. - (2 * inter / union)
    return loss


def detect_loss(y_true_cls, y_pred_cls, y_true_geo, y_pred_geo, training_mask):
    '''
    定义detect部分的损失函数
    :param y_true_cls:
    :param y_pred_cls:
    :param y_ture_gro:
    :param y_pred_geo:
    :param training_mask:
    :return:
    '''
    classification_loss = dice_coefficient(y_true_cls, y_pred_cls, training_mask)
    classification_loss *= 0.01

    # 顺次表示 top right bottom left
    d1_gt, d2_gt, d3_gt, d4_gt, theta_gt, = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
    d1_pred, d2_pred, d3_pred, d4_pred, theta_pred, = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)

    area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)  # 真实框的面积
    area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)  # 预测框的面积
    # 求两个面积的并集
    w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
    h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
    area_inter = w_union * h_union
    area_union = area_gt + area_pred - area_inter
    IOU = (area_inter + 1.0) / (area_union + 1.0)
    L_AABB = -tf.math.log(IOU+ 1e-10)
    L_theta = 1.0 - tf.math.cos(theta_pred - theta_gt)  # cos(0)=1
    L_g = L_AABB + 20.0 * L_theta
    loss_part = tf.reduce_mean(L_g * y_true_cls * training_mask)
    # print(loss_part)
    # print(classification_loss)
    return loss_part + classification_loss


def testloss(a, b, ):
    a = tf.reduce_mean(a[0])
    b = tf.reduce_mean(b[0])
    # c = tf.reduce_mean(c[0])
    return tf.reduce_mean([a, b])
