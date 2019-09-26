import tensorflow as tf
from keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, Dropout
from keras.layers import Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model

def make_discriminator(img_x_shape, img_y_shape, dropout=0, init_filters_n=64):

    def d_layer(layer_input, filters, f_size=4, bn=True, dropout_rate=0):
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        if dropout_rate:
            d = Dropout(dropout_rate)(d)
        return d

    img_A = Input(shape=img_x_shape)
    img_B = Input(shape=img_y_shape)

    combined_imgs = Concatenate(axis=-1)([img_A, img_B])

    d1 = d_layer(combined_imgs, init_filters_n, bn = False, dropout_rate=dropout)
    d2 = d_layer(d1, init_filters_n * 2, dropout_rate=dropout)
    d3 = d_layer(d2, init_filters_n * 4, dropout_rate=dropout)
    d4 = d_layer(d3, init_filters_n * 8, dropout_rate=dropout)
    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
    return Model([img_A, img_B], validity)

def make_generator(img_y_shape, dropout=0, init_filters_n=64):
    def conv2d(layer_input, filters, f_size=4, bn=True, dropout_rate=0):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        if dropout_rate:
            d = Dropout(dropout_rate)(d) 
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        u = BatchNormalization(momentum=0.8)(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = Concatenate()([u, skip_input])
        return u

    # Image input
    d0 = Input(shape=img_y_shape)

    # Downsampling
    d1 = conv2d(d0, init_filters_n, bn=False, dropout_rate=dropout)
    d2 = conv2d(d1, init_filters_n*2, dropout_rate=dropout)
    d3 = conv2d(d2, init_filters_n*4, dropout_rate=dropout)
    d4 = conv2d(d3, init_filters_n*8, dropout_rate=dropout)
    d5 = conv2d(d4, init_filters_n*8, dropout_rate=dropout)
    d6 = conv2d(d5, init_filters_n*8, dropout_rate=dropout)
    d7 = conv2d(d6, init_filters_n*8, dropout_rate=dropout)
    d8 = conv2d(d7, init_filters_n*8, dropout_rate=dropout)

    # Upsampling
    u0 = deconv2d(d8, d7, init_filters_n*8, dropout_rate=dropout)
    u1 = deconv2d(u0, d6, init_filters_n*8, dropout_rate=dropout)
    u2 = deconv2d(u1, d5, init_filters_n*8, dropout_rate=dropout)
    u3 = deconv2d(u2, d4, init_filters_n*8, dropout_rate=dropout)
    u4 = deconv2d(u3, d3, init_filters_n*4, dropout_rate=dropout)
    u5 = deconv2d(u4, d2, init_filters_n*2, dropout_rate=dropout)
    u6 = deconv2d(u5, d1, init_filters_n  , dropout_rate=dropout)

    u7 = UpSampling2D(size=2)(u6)
    output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

    return Model(d0, output_img)

