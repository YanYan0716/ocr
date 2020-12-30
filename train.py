import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.keras as keras

from Module.DetectLoss import detect_loss
from Module.RecognitionBackbone import Recognition_model
from Module.RecognitionLoss import recognition_loss
from Module.DetectBackbone import Detect_model
from DataPreprocess.DataGen import generator
import config

if __name__ == '__main__':
    MAX_EPOCHS = 1
    THETA = 0.01  # 控制检测和识别loss占总体loss的权重
    TRAIN = True
    CONTINUE_TRAIN = False
    MODEL_WEIGHTS_DIR = './model_weights/summary_weights/'
    SAVE_MODEL = False
    BEST_LOSS = 1000
    LOSS_STEP = 20  # 设置评估loss的步长
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    LEARNING_RATE = 0.0001

    # 构建数据库
    # ----通过tf.data.Dataset.from_generator产生输入数据
    dataset = tf.data.Dataset.from_generator(
        generator=generator,
        output_types=(
            tf.float32, tf.string, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.int32,
            tf.int32, tf.int32
        ),
    )

    # 搭建Detect网络
    weight_dir = './model_weights/efficientnetb0/efficientnetb0_notop.h5'
    detectmodel = Detect_model(base_weights_dir=weight_dir).model()

    # 搭建Recognition网络
    regmodel = Recognition_model(lstm_hidden_num=256).model()

    # 定义损失函数
    # detectloss = testloss()
    # regloss = recognition_loss()

    #  模型融合
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

    if TRAIN:
        for layer in detectmodel.layers:
            layer.trainable = False
        detectmodel.layers[-1].trainable = True
        regmodel.trainable = True
    else:
        detectmodel.trainable = False
        regmodel.trainable = False

    if CONTINUE_TRAIN:
        summary_model = keras.models.load_model(MODEL_WEIGHTS_DIR)

    print(len(summary_model.trainable_weights))

    optim = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # 训练过程
    for i in range(MAX_EPOCHS):
        print(f'EPOCH {i+1} for MAX_EPOCH {MAX_EPOCHS}...')
        temp_loss = 0
        for batch, (
                images, image_fns, score_maps, geo_maps, training_masks, transform_matrixes, boxes_masks, box_widths, \
                text_labels_sparse_0, text_labels_sparse_1, text_labels_sparse_2) in enumerate(dataset):
            with tf.GradientTape() as tape:
                input_box_masks = tf.expand_dims(boxes_masks, axis=0)
                input_box_widths = tf.expand_dims(box_widths, axis=0)
                input_box_info = tf.transpose(tf.concat([input_box_masks, input_box_widths], axis=0))

                shared_feature, f_score, f_geometry, recognition_logits = summary_model(
                    [images/255.0, transform_matrixes, input_box_info]
                )

                DetectLoss = detect_loss(tf.cast(score_maps, tf.float32),
                                         tf.cast(f_score, tf.float32),
                                         tf.cast(geo_maps, tf.float32),
                                         tf.cast(f_geometry, tf.float32),
                                         tf.cast(training_masks, tf.float32))

                RecognitionLoss = recognition_loss(recognition_logits,
                                                   text_labels_sparse_0,
                                                   text_labels_sparse_1,
                                                   text_labels_sparse_2, )

            total_loss = tf.cast(DetectLoss, dtype=tf.float32) + THETA * tf.cast(RecognitionLoss, dtype=tf.float32)

            grad = tape.gradient([DetectLoss, RecognitionLoss], summary_model.trainable_weights)
            # # 观察是否可以进行反向传播
            # j = 0
            # for i in range(len(grad)):
            #     if hasattr(grad[i], 'shape'):
            #         j += 1
            #         print(i, j, grad[i].shape)
            optim.apply_gradients(zip(grad, summary_model.trainable_weights))
            # summary_model.save_weights(MODEL_WEIGHTS_DIR)
            # # 计算平均loss并保存模型
            temp_loss += total_loss.numpy()
            if (batch + 1) % LOSS_STEP == 0:
                now_loss = temp_loss / LOSS_STEP
                print(f'the loss is :{now_loss}')
                if now_loss < BEST_LOSS:
                    print(f'saving model to ----> {MODEL_WEIGHTS_DIR}')
                    BEST_LOSS = now_loss
                    summary_model.save_weights(MODEL_WEIGHTS_DIR+str(i))
                temp_loss = 0
            break
        break
