import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import math


from DataPreprocess.DataGen import get_project_matrix_and_width
from Module.DetectBackbone import Detect_model
from Module.RecognitionBackbone import Recognition_model
from DetectUtil import detect_contours, sort_poly
import config
from Module.RecognitionLoss import decode


def ground_truth_to_word(recog_decode):
    text_info = []
    for i in range(len(recog_decode)):
        str_index = recog_decode[i]
        if str_index != -1:
            if i == 94:
                continue
            text_info += (config.CHAR_VECTOR[str_index])
    return text_info


if __name__ == '__main__':
    '''
    本代码中图像尺寸为512*512 在实际场景中图像的长宽比例不同需要进行相应的缩放
    '''
    # 获取一张图片
    img = 'img.jpg'

    # 限制图片的长宽尺寸，保持原比例
    img = cv2.imread(img)

    # 交换颜色通道
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tf.convert_to_tensor(img, dtype=tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)
    # print(f'img shape: {img.shape}')
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
    # print(summary_model.summary())
    # for i in range(len(summary_model.layers[4].inputs)):
    #     print(i)
    #     print(summary_model.layers[4].inputs[i].shape)

    img = np.random.random((1, 512, 512, 3))
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    input_box_mask = []
    with tf.GradientTape() as tape:
        # 经过detect的部分
        x = summary_model.layers[0](img)
        out = summary_model.layers[0](x)
        shared_feature = out[0]
        f_score = out[1]
        f_geometry = out[2]
        # 处理detect的结果
        boxes = detect_contours(score_map=f_score, geo_map=f_geometry)

        # 处理recognition部分
        if boxes is not None and boxes.shape[0] != 0:
            # 预处理，获得进入Recognition部分的信息
            input_roi_boxes = []
            roi_boxes = boxes[:, :8].reshape((-1, 4, 2))
            for box in boxes:
                box = sort_poly(box)
                input_roi_boxes.append(box.reshape(1, 8)[0])
            input_roi_boxes = np.array(input_roi_boxes)
            boxes_masks = [0] * input_roi_boxes.shape[0]
            transform_matrixes, box_widths = get_project_matrix_and_width(input_roi_boxes)
            sum_step = math.floor((boxes.shape[0]+config.RECOG_BATCH-1)/config.RECOG_BATCH)
            recod_decode_list =[]
            for step in range(sum_step):
                start_index = step*config.RECOG_BATCH
                if step == sum_step-1:
                    end_index = boxes.shape[0]
                else:
                    end_index = (step+1)*config.RECOG_BATCH
                # 整合Recognition需要的信息
                shared_feature = shared_feature
                input_transform_matrix = transform_matrixes[start_index:end_index]
                input_box_widths = box_widths[start_index:end_index]
                input_box_mask[0] = boxes_masks[start_index:end_index]

                input_box_masks = tf.expand_dims(input_box_masks, axis=0)
                input_box_widths = tf.expand_dims(input_box_widths, axis=0)
                input_box_info = tf.transpose(tf.concat([input_box_masks, input_box_widths], axis=0))

                # 经过识别部分
                logits = summary_model.layers[4]([shared_feature, transform_matrixes, input_box_info])
                # 将识别结果解码
                dense_decode = decode(logits=logits, seq_len=input_box_widths)
                recod_decode_list.append(dense_decode)
        # -----------------汇总全部结果
        with open(config.TEST_RESULT_PATH, 'w') as f:
            for i, box in enumerate(boxes):
                box = sort_poly(box.astype(np.int32))
                if np.linalg.norm(box[0]-box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                    continue
                reg_result = ground_truth_to_word(recod_decode_list[i])
                print('{},{},{},{},{},{},{},{},{}\r'.format(
                    box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
                    reg_result))
                f.write('{},{},{},{},{},{},{},{},{}\r\n'.format(
                    box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
                    reg_result))
                # 此处省略检测框滑道原图的可视化部分

