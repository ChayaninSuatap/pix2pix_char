from PIL import Image, ImageOps
import random

def flip_left_right(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def flip_up_down(img):
    return img.transpose(Image.FLIP_TOP_BOTTOM)

def rotate(img, angle, invert_color=False):
    if not invert_color:
        img = ImageOps.invert(img)
    img = img.rotate(angle=angle, expand=1)
    if not invert_color:
        img = ImageOps.invert(img)
    return img

def augment_img(x_img, y_img, img_x_resize, img_y_resize, img_x_resize_final=None, img_y_resize_final=None, invert_color=False, scale=False):
    if img_x_resize_final is None: img_x_resize_final = img_x_resize
    if img_y_resize_final is None: img_y_resize_final = img_y_resize

    #flip
    if random.randint(0, 1) == 0:
        x_img = flip_left_right(x_img)
        y_img = flip_left_right(y_img)
    if random.randint(0, 1) == 0:
        x_img = flip_up_down(x_img)
        y_img = flip_up_down(y_img)
    #rotate
    angle = random.randint(0, 360)
    x_img = rotate(x_img, angle, invert_color=invert_color).resize(img_x_resize)
    y_img = rotate(y_img, angle, invert_color=invert_color).resize(img_y_resize)
    #scale
    if scale:
        scale_factor = random.uniform(0.5, 1.2)
        scale_size = (int(img_x_resize[0] * scale_factor), int(img_x_resize[1] * scale_factor))
        scaled_patch_x = x_img.resize(scale_size)
        scaled_patch_y = y_img.resize(scale_size)
        #random where to paste patch
        #create empty image
        x_img= Image.new('L', img_x_resize_final, (0,) if invert_color else (255,))
        y_img= Image.new('L', img_y_resize_final, (0,) if invert_color else (255,))

        try:
            x_pos = random.randrange(0, scaled_patch_x.width - x_img.width)
            x_crop_bound = x_pos + x_img.width
        except:
            x_pos = 0
            x_crop_bound = scaled_patch_x.width
        try:
            y_pos = random.randrange(0, scaled_patch_x.height - x_img.height)
            y_crop_bound = y_pos + x_img.height
        except:
            y_pos = 0
            y_crop_bound = scaled_patch_x.height
        

        x_crop = scaled_patch_x.crop((x_pos, y_pos, x_crop_bound, y_crop_bound))
        y_crop = scaled_patch_y.crop((x_pos, y_pos, x_crop_bound, y_crop_bound))

        x_paste_pos = 0 if x_img.width == x_crop.width else random.randrange(0, x_img.width - x_crop.width)
        y_paste_pos = 0 if x_img.height == x_crop.height else random.randrange(0, x_img.height - x_crop.height)
        x_img.paste(scaled_patch_x, (x_paste_pos, y_paste_pos))
        y_img.paste(scaled_patch_y, (x_paste_pos, y_paste_pos))

    return x_img, y_img

if __name__ == '__main__':
    img = Image.open('x_path/4_1.jpg')
    img = rotate(img, 59)

    img.show()
    input()