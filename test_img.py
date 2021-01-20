import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import tensorflow as tf
import tensorflow.keras as keras


from Module.DetectBackbone import Detect_model
from Module.RecognitionBackbone import Recognition_model


if __name__ == '__main__':
    # 获取一张图片
    img = 'img.jpg'

    # 限制图片的长宽尺寸，保持原比例
    img = cv2.imread(img)

    # 交换颜色通道
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tf.convert_to_tensor(img, dtype=tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)
    print(f'img shape: {img.shape}')
    # ------------------
    WEIGHT_DIR = './model_weights/efficientnetb0/efficientnetb0_notop.h5'
    MODEL_WEIGHTS_DIR = '/content/drive/MyDrive/tensorflow/ocr/ocr/model_weights/summary_weights/best'

    # 搭建Detect网络
    detectmodel = Detect_model(base_weights_dir=WEIGHT_DIR).model()

    # 搭建Recognition网络
    regmodel = Recognition_model(lstm_hidden_num=256).model()

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

    detectmodel.trainable = False
    regmodel.trainable = False
    summary_model.load_weights(MODEL_WEIGHTS_DIR)

    # 预测结果

    shared_feature, f_score, f_geometry, recognition_logits = summary_model(
        [images / 255.0, transform_matrixes, input_box_info]
    )



    # #旧方法--------------------------
    # # 获取一张图片
    # img = 'img.jpg'
    #
    # # 限制图片的长宽尺寸，保持原比例
    # img = cv2.imread(img)
    # img, (ratio_h, ratio_w) = ResizeImg(img)
    #
    # # 交换颜色通道
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = tf.convert_to_tensor(img, dtype=tf.uint8)
    # img = tf.expand_dims(img, axis=0)
    # print(f'img shape: {img.shape}')
    #
    # # 网络预测过程
    # weight_dir = './model_weights/efficientnetb0/efficientnetb0_notop.h5'
    # detectmodel = DetectModel()
    # share_data, score, geometry = detectmodel.detect_eval_model(weights_dir=weight_dir,
    #                                                             model_name='EfficientNetB0',
    #                                                             imgs=img)
    #
    # print(share_data.shape)
    # print(score.shape)
    # print(geometry.shape)
