# coding:utf-8
CHAR_VECTOR = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-~`<>'.:;^/|!?$%#@&*()[]{}_+=,\\\""
NUM_CLASSES = len(CHAR_VECTOR) + 1
MIN_TEXT_SIZE = 10  #确定最小的字大小，小于这个尺寸的就扔掉
BATCH_SIZE = 1
BLACK_INDEX = len(CHAR_VECTOR)
RECOG_BATCH = 12
TEST_RESULT_PATH = './result.txt'
