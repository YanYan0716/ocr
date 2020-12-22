import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import tensorflow as tf
import tensorflow.keras as keras

from Module.DetectLoss import detect_loss
from Module.RecognitionBackbone import Recognition_model
from Module.RecognitionLoss import recognition_loss
from Module.DetectBackbone import Detect_model
from Module.RoiRotate import RoiRotate
from Module.CustomFit import CustomFit

if __name__ == '__main__':
    # 构建数据库

    # 获取一张图片
    img = 'img.jpg'

    # 限制图片的长宽尺寸，保持原比例
    img = cv2.imread(img)

    # 交换颜色通道
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tf.convert_to_tensor(img, dtype=tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)
    print(f'img shape: {img.shape}')

    # 搭建Detect网络
    weight_dir = './model_weights/efficientnetb0/efficientnetb0_notop.h5'
    detectmodel = Detect_model(trainable=False, base_weights_dir=weight_dir).model()
    # 加入roi_rotate
    roi_rotate = RoiRotate()
    # 搭建Recognition网络
    regmodel = Recognition_model(lstm_hidden_num=256).model()

    # 定义损失函数
    detectloss = detect_loss()
    regloss = recognition_loss()

    # training = CustomFit(detectmodel, roi_rotate, regmodel)
    #
    # training.compile(
    #     optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    #     detectloss=detectloss,
    #     regloss=regloss,
    # )

    # 训练过程
    NUM_EPOCHS = 10
    for epoch in range(NUM_EPOCHS):
        print('start of training epoch {epoch}')
        for batch_idx, (x_batch, y_batch) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                pass








    # for i in range(len(detectmodel.layers)):
    #     print(detectmodel.layers[i])
    # for j in range(4):
    x = detectmodel.predict_step(img)
    for i in range(len(x)):
        print(x[i].shape)
