# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt



import os
import numpy as np

from pathlib import Path


#ML library
from sklearn.decomposition import PCA
import PIL
from PIL import Image
import tensorflow as tf

#step 1 denoise, step 2 edge det then draw a box
#we are using edge detection using sobel operators for edge detection because of speed concerns.
#here are the x and y ops
#consts
SOBEL_X = (1/8)*np.array([[-1, 0, 1], [-2, 0, 2],[-1, 0, 1]])
SOBEL_Y = (1/8)*np.array([[1, 2, 1],[0, 0 ,0],[-1, -2, -1]])

NORM = 36*36

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

#crops images into data that can be fed into the machine learning algorithm.
def crop_data(dir, filename, bit, setname= "train"):
    ctr = 0
    file = open(dir + filename)
    dataset_x36 = np.empty((0,1296),int)


    for line in file.readlines():
        #ctr = 0
        image_name, numbers = line.split(".")
        #print(f"image_name {image_name} nums {numbers}")
        n_arr = numbers.split(maxsplit=1)
        image_name = image_name + '.' + n_arr[0]
        #n_arr = [int (x) for x in n_arr[1].split()]
        np_arr = np.fromstring(n_arr[1],dtype=int, sep=' ')



        print(f"image_name {image_name} nums {np_arr} 1 {np_arr[1]} 2 {np_arr[2]} 3 {np_arr[3]} 4 {np_arr[4]}")

        img = Image.open(dir + f"{setname}_images_{bit}_bit/" + image_name)



        width, height = img.size

        #cropped = img.crop((np_arr[1],  np_arr[3],  np_arr[2] ,  np_arr[4]))

        left = np_arr[1]
        right = np_arr[2]
        top = height - np_arr[3]
        bottom = height - np_arr[4]

        #error checking
        if left > right:
            tmp = left
            left = right
            right = tmp

        if top > bottom:
            tmp = top
            top = bottom
            bottom = tmp

        rows = right - left
        cols = bottom - top
        dims = rows * cols





        if top != bottom and left != right and dims >= NORM:


            #img_np = np.asarray(img)

            #norm_vec = normalize(img_np)

            cropped = img.crop((left, top, right, bottom))
            # dimensionality reduction
            cropped = cropped.resize((36, 36))

            crop_np = np.asarray(cropped)
            crop_np = crop_np.flatten()



            dataset_x36 = np.vstack((dataset_x36, crop_np))


            if not os.path.exists(f"output/{bit}_bit/boxes/"):
                Path(f"output/{bit}_bit/boxes/").mkdir(parents=True,exist_ok=True)

            cropped = cropped.save(f"output/{bit}_bit/boxes/{np_arr[0]}_{ctr}_{image_name}")

            ctr += 1
            #print(f"normvec {norm_vec}")

        if ctr > 1024: break #prevents from executing for all images when testing

    return dataset_x36





def train(data):
    pass

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    d_x8 = crop_data("8_bit_dataset/8_bit_dataset/", "train_labels_8_bit.txt", 8 )
    d_x16 = crop_data("16_bit_dataset/16_bit_dataset/", "labels_train_16_bit.txt", 16 )

    #normalize
    normd_x8 = d_x8 - np.mean(d_x8, axis=0, keepdims=True)
    normd_x16 = d_x16 - np.mean(d_x16, axis=0, keepdims=True)

    print(normd_x8.shape)

    #pca to a smaller number
    #testing with 256 of 1024 but would probably be 1024 out of 10000 images
    reduced = PCA(n_components=256)
    rm_8 = reduced.fit_transform(normd_x8)
    rm_16 = reduced.fit_transform(normd_x16)

    #save to text
    np.savetxt("x8_train.txt", rm_8)
    np.savetxt("x16_train.txt", rm_16)




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
