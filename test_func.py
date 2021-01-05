from DataPreprocess.DataGen import load_annoatation
import numpy as np


def func(a, b):
    c = a+b

    c+=1
    return c


if __name__ == '__main__':
    a=func(1, 2)
    print(a)