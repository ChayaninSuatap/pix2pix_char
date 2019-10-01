import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
from augment_util import augment


def read_img(path, size):
    img = Image.open(path)
    img = img.resize(size)
    img = np.asarray(img) / 127.5 - 1
    return img

def load_sample_data(img_size):
    imgs = []
    labels = []
    for fn in os.listdir('test/sample_x'):
        img = Image.open('test/sample_x/' + fn)
        img = img.resize(img_size)
        img = np.asarray(img) / 127.5 - 1
        label = int(fn[2:-4]) - 1
        imgs.append(img)
        labels.append(label)
    return imgs, labels

def make_dataset_cache(img_x_resize=(40,40), img_y_resize=(40,40)):
    x_path = 'datasets/x_chars/'
    y_path = 'datasets/y_chars/'

    #load all data
    x_imgs = []
    for fn in os.listdir(x_path):
        img = Image.open(x_path + fn)
        img = img.resize(img_x_resize)
        label = int(fn[2:-4]) - 1
        x_imgs.append( (img, label))
    
    random.shuffle(x_imgs)
    y_imgs = []        
    for foldername in os.listdir(y_path):
        for fn in os.listdir(y_path+foldername):
            label = int(foldername) - 1
            img = Image.open(y_path+foldername+'/'+fn)
            img = img.resize(img_y_resize)
            y_imgs.append( (img, label))

    return x_imgs, y_imgs


def make_dataset_generator(batch_size, dataset_cache, img_x_resize=(40,40), img_y_resize=(40,40), use_label=False):
    x_imgs, y_imgs = dataset_cache 
    random.shuffle(x_imgs)

    y_img_dict = {}
    for img, label in y_imgs:
        if label not in y_img_dict:
            y_img_dict[label] = [img]
        else:
            y_img_dict[label].append(img)

    # yielding process
    x_img_i = 0
    img_x_chrunk = []
    img_y_chrunk = []
    label_chrunk = []
    while True:
        x_img = x_imgs[x_img_i][0]
        label = x_imgs[x_img_i][1] #0=img, 1=label
        y_img_i = random.randint(0, len(y_img_dict[label]) - 1)
        y_img = y_img_dict[label][y_img_i]

        #augment
        x_img , y_img = augment(x_img, y_img, img_x_resize, img_y_resize)
        #asarray to normalize
        x_img = np.asarray(x_img) / 127.5 - 1
        y_img = np.asarray(y_img) / 127.5 - 1
        
        img_x_chrunk.append( x_img.reshape(img_x_resize[0], img_x_resize[1], 1))
        img_y_chrunk.append( y_img.reshape(img_y_resize[0], img_y_resize[1], 1))
        label_chrunk.append( label)
        x_img_i += 1

        if x_img_i == len(x_imgs) or len(img_x_chrunk) == batch_size:
            if not use_label:
                yield np.asarray(img_x_chrunk), np.asarray(img_y_chrunk)
            elif use_label:
                yield np.asarray(img_x_chrunk), np.asarray(img_y_chrunk), np.asarray(label_chrunk)
            img_x_chrunk = []
            img_y_chrunk = []
            label_chrunk = []

            if x_img_i == len(x_imgs):
                break

if __name__ == '__main__':
    count = 0
    for x_imgs, y_imgs in make_dataset_generator(32):
        count+= len(x_imgs)
    print(count)


