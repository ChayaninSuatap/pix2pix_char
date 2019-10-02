from PIL import Image, ImageOps
import random

def flip_left_right(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)
    

def flip_up_down(img):
    return img.transpose(Image.FLIP_TOP_BOTTOM)

def rotate(img, angle):
    img = ImageOps.invert(img)
    img = img.rotate(angle=angle, expand=1)
    img = ImageOps.invert(img)
    return img

def augment(x_img, y_img, img_x_resize, img_y_resize):
    if random.randint(0, 1) == 0:
        x_img = flip_left_right(x_img)
        y_img = flip_left_right(y_img)
    if random.randint(0, 1) == 0:
        x_img = flip_up_down(x_img)
        y_img = flip_up_down(y_img)
    # angle = random.randint(0, 360)
    # x_img = rotate(x_img, angle).resize(img_x_resize)
    # y_img = rotate(y_img, angle).resize(img_y_resize)

    return x_img, y_img

if __name__ == '__main__':
    img = Image.open('x_path/4_1.jpg')
    img = rotate(img, 59)

    img.show()
    input()