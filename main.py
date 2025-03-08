# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt

import numpy as np

#ML library
import PIL
import tensorflow as tf

#step 1 denoise, step 2 edge det then draw a box
#we are using edge detection using sobel operators for edge detection because of speed concerns.
#here are the x and y ops
SOBEL_X = (1/8)*np.array([[-1, 0, 1], [-2, 0, 2],[-1, 0, 1]])
SOBEL_Y = (1/8)*np.array([[1, 2, 1],[0, 0 ,0],[-1, -2, -1]])

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

#crops images into data that can be fed into the machine learning algorithm.
def crop_data():
    pass

def train(data):
    pass

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
