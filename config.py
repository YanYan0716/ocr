# coding:utf-8
CHAR_VECTOR = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-~`<>'.:;^/|!?$%#@&*()[]{}_+=,\\\""
NUM_CLASSES = len(CHAR_VECTOR) + 1
MIN_TEXT_SIZE = 400  #不知道用途，没有用到
BATCH_SIZE = 1
BLACK_INDEX = len(CHAR_VECTOR)