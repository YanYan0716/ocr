import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


from Module.DetectLoss import detect_loss
from Module.RecognitionBackbone import Recognition_model
from Module.RecognitionLoss import recognition_loss
from Module.DetectBackbone import Detect_model
from Module.RoiRotate import RoiRotate
from DataPreprocess.DataGen import generator
import config

if __name__ == '__main__':
    MAX_EPOCHS = 2
    THETA = 0.9
    TRAIN = True
    CONTINUE_TRAIN = False
    DETECT_WEIGHTS_DIR = './model_weights/detect_weights'
    REG_WEIGHTS_DIR = './model_weights/reg_weights'
    # 构建数据库
    # ----通过tf.data.Dataset.from_generator产生输入数据
    dataset = tf.data.Dataset.from_generator(
        generator=generator,
        output_types=(
            tf.float32, tf.string, tf.int32, tf.int32, tf.int32, tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32,
        ),
    )

    # 搭建Detect网络
    weight_dir = './model_weights/efficientnetb0/efficientnetb0_notop.h5'
    detectmodel = Detect_model(base_weights_dir=weight_dir).model()

    # 加入roi_rotate
    roi_rotate = RoiRotate()

    # 搭建Recognition网络
    regmodel = Recognition_model(lstm_hidden_num=256).model()

    # 定义损失函数
    # detectloss = detect_loss()
    # regloss = recognition_loss()

    # 判断是否训练
    if TRAIN == True:
        for layer in detectmodel.layers:
            layer.trainable = False
        detectmodel.layers[-1].trainable = True
        regmodel.trainable = True
    else:
        detectmodel.trainable = False
        regmodel.trainable = False

    # 训练过程
    for i in range(1):
        for batch, (images, image_fns, score_maps, geo_maps, training_masks, transform_matrixes, boxes_masks, box_widths, \
                    text_labels_sparse_0, text_labels_sparse_1, text_labels_sparse_2) in enumerate(dataset):
            with tf.GradientTape() as tape:
                shared_feature, f_score, f_geometry = detectmodel(images)
                recognition_logits = regmodel(shared_feature,
                                              transform_matrixes,
                                              boxes_masks,
                                              box_widths)

                DetectLoss = detect_loss(score_maps,
                                         tf.cast(f_score, tf.int32),
                                         geo_maps,
                                         tf.cast(f_geometry, tf.int32),
                                         tf.cast(training_masks, tf.int32))
                RecognitionLoss = recognition_loss(recognition_logits,
                                          text_labels_sparse_0,
                                          text_labels_sparse_1,
                                          text_labels_sparse_2,)

                total_loss = DetectLoss + THETA * tf.cast(RecognitionLoss, dtype=tf.float64)
                # 反向传播
                # grad_detect = tape.gradient(DetectLoss, detectmodel.trainable_weights)
                # grad_reg = tape.gradient(RecognitionLoss, regmodel.trainable_weights)

                print(total_loss)
                break
            break
        break
