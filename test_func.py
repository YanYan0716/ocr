import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append('D:\\algorithm\\ocr')
sys.path.append('E:\\algorithm\\ocr')
sys.path.append('/home/epai/yanqian/ocr')
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

from Module.DetectLoss import detect_loss
from Module.RecognitionBackbone import Recognition_model
from Module.RecognitionLoss import recognition_loss
from Module.DetectBackbone import Detect_model
from DataPreprocess.DataGen import generator
from Module.WarmupLR import WarmUpLR


if __name__ == '__main__':
    MAX_EPOCHS = 10000
    THETA = 0.01  # 控制检测和识别loss占总体loss的权重
    TRAIN = True
    CONTINUE_TRAIN = True
    SAVE_MODEL = False
    BEST_LOSS = 1000
    LOSS_STEP = 1  # 设置评估loss的步长

    LEARNING_RATE = 0.00001
    WEIGHT_DIR = './model_weights/efficientnetb0/efficientnetb0_notop.h5'
    MODEL_WEIGHTS_DIR = './model_weights/summary_weights/best'

    # 搭建Detect网络
    detectmodel = Detect_model(base_weights_dir=WEIGHT_DIR).model()

    # 搭建Recognition网络
    regmodel = Recognition_model(lstm_hidden_num=256).model()

    inputs1 = detectmodel.layers[0].input
    inputs2 = detectmodel(inputs1)[0]
    inputs3 = regmodel.inputs[1]
    inputs4 = regmodel.inputs[2]
    all_inputs = [inputs1, inputs3, inputs4]

    outputs0 = detectmodel(inputs1)[0]
    outputs1 = detectmodel(inputs1)[1]
    outputs2 = detectmodel(inputs1)[2]
    all_outputs = [outputs0, outputs1, outputs2, regmodel([outputs0, inputs3, inputs4])]

    summary_model = keras.Model(all_inputs, all_outputs)

    for layer in detectmodel.layers:
        layer.trainable = True
    regmodel.trainable = True

    print(len(summary_model.trainable_weights))

    optim = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    data = np.load('/content/ocr/test_data.npy', allow_pickle=True)
    img = data[0]
    score_maps = data[1]
    geo_maps = data[2]
    training_masks = data[3]
    transform_matrixes = data[4]
    boxes_masks = data[5]
    box_widths = data[6]
    text_labels_sparse_0 = data[7]
    text_labels_sparse_1 = data[8]
    text_labels_sparse_2 = data[9]

    '''
    np.random.seed(1)
    img = np.random.random((1, 512, 512, 3))
    img = tf.convert_to_tensor(img, dtype=tf.float32)

    score_maps = np.random.random((1, 128, 128, 1))
    score_maps = tf.convert_to_tensor(score_maps, dtype=tf.float32)

    training_masks = np.random.random((1, 128, 128, 1))
    training_masks = tf.convert_to_tensor(training_masks, dtype=tf.float32)

    transform_matrixes = np.random.random((2, 6))
    transform_matrixes = tf.convert_to_tensor(transform_matrixes, dtype=tf.float32)

    geo_maps = np.random.random((1, 128, 128, 5))
    geo_maps = tf.convert_to_tensor(geo_maps, dtype=tf.float32)

    boxes_masks = np.array([0., 0.])
    boxes_masks = tf.convert_to_tensor(boxes_masks, dtype=tf.float32)

    box_widths = np.array([17., 18.])
    box_widths = tf.convert_to_tensor(box_widths, tf.float32)

    text_labels_sparse_0 = np.array([
        [0,0], [0,1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2]
    ])
    text_labels_sparse_0 = tf.convert_to_tensor(text_labels_sparse_0, tf.int32)

    text_labels_sparse_1 = np.array([12, 12, 2, 2, 11, 30, 10])
    text_labels_sparse_1 = tf.convert_to_tensor(text_labels_sparse_1, tf.int32)

    text_labels_sparse_2 = np.array([2, 4])
    text_labels_sparse_2 = tf.convert_to_tensor(text_labels_sparse_2, tf.int32)
    '''

    # 训练过程
    for i in range(MAX_EPOCHS):
        with tf.GradientTape() as tape:
            input_box_masks = tf.expand_dims(boxes_masks, axis=0)
            input_box_widths = tf.expand_dims(box_widths, axis=0)
            input_box_info = tf.transpose(tf.concat([input_box_masks, input_box_widths], axis=0))
            shared_feature, f_score, f_geometry, recognition_logits = summary_model(
                    [img, transform_matrixes, input_box_info]
                )
            DetectLoss = detect_loss(tf.cast(score_maps, tf.float32),
                                     tf.cast(f_score, tf.float32),
                                     tf.cast(geo_maps, tf.float32),
                                     tf.cast(f_geometry, tf.float32),
                                     tf.cast(training_masks, tf.float32))

            RecognitionLoss = recognition_loss(recognition_logits,
                                               text_labels_sparse_0,
                                               text_labels_sparse_1,
                                               text_labels_sparse_2,)

            RecognitionLoss = THETA * tf.cast(RecognitionLoss, dtype=tf.float32)
            DetectLoss = tf.cast(DetectLoss, tf.float32)
            total_loss = DetectLoss + RecognitionLoss
            print(f'[epoch {i}/ MAXEPOCH {MAX_EPOCHS}]:[det: %.5f]'%DetectLoss+' [reg: %.5f]/'%RecognitionLoss+'[total: %.5f'%total_loss+']')

        grad = tape.gradient([total_loss], summary_model.trainable_weights)

        grad = [tf.clip_by_norm(g, 2) for g in grad]
        optim.apply_gradients(zip(grad, summary_model.trainable_weights))