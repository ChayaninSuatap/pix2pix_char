import os
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np
import random
from augment_util import augment_img

def read_img(path, size, invert_color=False):
    img = Image.open(path)
    img = img.resize(size)
    # if invert_color:
        # img = ImageOps.invert(img)
    img = np.asarray(img)
    if invert_color:
        img = 255 - img
    img = img / 127.5 - 1
    return img

def load_sample_data(img_size, invert_color=False):
    imgs = []
    labels = []
    for fn in os.listdir('test/sample_x'):
        img = Image.open('test/sample_x/' + fn)
        img = img.resize(img_size)
        if invert_color:
            img = ImageOps.invert(img)
        img = np.asarray(img) / 127.5 - 1
        label = int(fn[2:-4]) - 1
        imgs.append(img)
        labels.append(label)
    return imgs, labels

def make_dataset_cache(img_x_resize=(40,40), img_y_resize=(40,40), invert_color=False):
    x_path = 'datasets/x_chars/'
    y_path = 'datasets/y_chars/'

    #load all data
    x_imgs = []
    for fn in os.listdir(x_path):
        img = Image.open(x_path + fn)
        img = img.resize(img_x_resize)
        if invert_color:
            img = ImageOps.invert(img.convert('L'))
            img = img.convert('P')
        label = int(fn[2:-4]) - 1
        x_imgs.append( (img, label))
    
    random.shuffle(x_imgs)
    y_imgs = []        
    for foldername in os.listdir(y_path):
        for fn in os.listdir(y_path+foldername):
            label = int(foldername) - 1
            img = Image.open(y_path+foldername+'/'+fn)
            img = img.resize(img_y_resize)
            if invert_color:
                img = ImageOps.invert(img)
            y_imgs.append( (img, label))

    return x_imgs, y_imgs

def make_dataset_generator(batch_size, dataset_cache, img_x_resize=(40,40), img_y_resize=(40,40),
    img_x_resize_final=None, img_y_resize_final=None,
    use_label=False,
    invert_color=False, augment=True, scale=False):

    if img_x_resize_final is None: img_x_resize_final = img_x_resize
    if img_y_resize_final is None: img_y_resize_final = img_y_resize

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
        if augment:
            x_img , y_img = augment_img(x_img, y_img, img_x_resize, img_y_resize, img_x_resize_final=img_x_resize_final, img_y_resize_final=img_y_resize_final,
             invert_color=invert_color, scale=scale)

        #check img
        # check_img_fn = str(random.randint(0,999999))
        # x_img.save('check_img/' + check_img_fn + 'x.jpg')
        # y_img.save('check_img/' + check_img_fn + 'y.jpg')
        
       #asarray to normalize
        x_img = np.asarray(x_img) / 127.5 - 1
        y_img = np.asarray(y_img) / 127.5 - 1

        # print('before reshape shape : ',x_img.shape)
        # input()
        
        img_x_chrunk.append( x_img.reshape(img_x_resize_final[1], img_x_resize_final[0], 1))
        img_y_chrunk.append( y_img.reshape(img_y_resize_final[1], img_y_resize_final[0], 1))
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

def make_x_scale(sample_fn, save_path, img_x_resize):
    from augment_util import rotate
    import random

    for angle in range(0, 360, 30):
        for scale_factor in range(5,12):
            x_img = Image.open(sample_fn).convert('RGB')
            x_img = rotate(x_img, angle ).resize(img_x_resize)
            if scale_factor == 10:
                continue
            scale_factor /= 10
            scale_size = (int(img_x_resize[0] * scale_factor), int(img_x_resize[1] * scale_factor))
            scaled_patch_x = x_img.resize(scale_size)

            #create empty image
            x_img= Image.new('L', img_x_resize,  (255,))

            if scale_factor >= 1:
                x_pos = random.randrange(0, scaled_patch_x.width - x_img.width)
                y_pos = random.randrange(0, scaled_patch_x.width - x_img.width)
                x_img = scaled_patch_x.crop((x_pos, y_pos, x_img.width+x_pos, x_img.width+y_pos))

            else: #random position to paste
                x_pos = random.randrange(0, x_img.width - scaled_patch_x.width)
                y_pos = random.randrange(0, x_img.width - scaled_patch_x.width)
                x_img.paste(scaled_patch_x, (x_pos, y_pos))
                x_img.save(save_path+'/%.1f_%d.bmp' %(scale_factor, angle) )

if __name__ == '__main__':
    # count = 0
    # cache = make_dataset_cache(img_x_resize=(64,64), img_y_resize=(64,64), invert_color=True)
    # for x_imgs, y_imgs in make_dataset_generator(32, cache, img_x_resize=(64,64), img_y_resize=(64,64), invert_color=True, scale=True):
    #     count+= len(x_imgs)
    # print(count)
    make_x_scale('x_path/10_0.bmp', 'x_scale_4', (128, 128))


