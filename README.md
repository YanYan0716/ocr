# ocr
源码位置：

https://github.com/Pay20Y/FOTS_TF/tree/dev
https://github.com/Pay20Y/FOTS_TF

tensorflow=2.3.0

##
其中仍不太明白的地方：transformer

和原文不同的地方：

检测部分的基础网络结构

检测部分的unpool部分，原文中使用双线性插值将图片放大两倍本代码使用反卷积，
详见Module.DetectBackbone的第96~106行代码

训练：train.py
测试：test_img.py 对一张图片的测试

除主要代码，附加部分为对相关代码的理解



                