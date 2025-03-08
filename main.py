# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt



import os
import numpy as np


#ML library
import PIL
from PIL import Image
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
def crop_data(dir, filename, bit):
    file = open(dir + filename)
    for line in file.readlines():
        image_name, numbers = line.split(".")
        #print(f"image_name {image_name} nums {numbers}")
        n_arr = numbers.split(maxsplit=1)
        image_name = image_name + '.' + n_arr[0]
        #n_arr = [int (x) for x in n_arr[1].split()]
        np_arr = np.fromstring(n_arr[1],dtype=int, sep=' ')


        print(f"image_name {image_name} nums {np_arr}")

        img = Image.open(dir + f"train_images_{bit}_bit/" + image_name)







def train(data):
    pass

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    crop_data("8_bit_dataset/8_bit_dataset/", "train_labels_8_bit.txt", 8)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
