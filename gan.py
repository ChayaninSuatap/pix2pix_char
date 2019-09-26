import tensorflow as tf
from keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, Dropout
from keras.layers import Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
from dataset_util import make_dataset_generator, load_sample_data
import datetime
import os
import matplotlib.pyplot as plt

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

def make_generator(img_y_shape, dropout=0, init_filters_n=64, channels=1):
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

    # Upsampling
    u1 = deconv2d(d3, d2, init_filters_n*4, dropout_rate=dropout)
    u2 = deconv2d(u1, d1, init_filters_n*2, dropout_rate=dropout)

    u7 = UpSampling2D(size=2)(u2)
    output_img = Conv2D(channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)
    model = Model(d0, output_img)
    return model

def make_gan(img_x_shape, img_y_shape, dis_dropout, gen_dropout):
    optimizer = Adam(0.0002, 0.5)

    dis = make_discriminator(img_x_shape, img_y_shape, dis_dropout)
    dis.compile(loss='mse',
        optimizer = optimizer,
        metrics=['accuracy'])
    
    gen = make_generator(img_y_shape, gen_dropout)
    x_input_layer = Input(shape=img_x_shape)
    gen_output_layer = gen(x_input_layer)

    dis.trainable = False
    dis_output_layer = dis([gen_output_layer, x_input_layer])

    gan = Model(inputs=x_input_layer, outputs=[dis_output_layer, gen_output_layer])
    gan.compile(loss=['mse', 'mae'],
        loss_weights=[1, 100],
        optimizer=optimizer)
    
    return gan, gen, dis

def train(gan, gen, dis, img_x_size, img_y_size, epochs, batch_size,
    init_epoch=1,
    save_weights_each_epochs=1,
    save_weights_checkpoint_each_epochs=5,
    save_weights_path=''):
    start_time = datetime.datetime.now()

    patch = int(img_x_size[1] / 2**4)
    disc_patch = (patch, patch, 1)
    valid_label = np.ones((batch_size,) + disc_patch)
    fake_label = np.zeros((batch_size,) + disc_patch)

    d_losses = []
    g_losses = []

    for epoch in range(init_epoch, epochs+1):
        batch_i = 0
        for x_imgs, y_imgs in make_dataset_generator(batch_size, img_x_size, img_y_size):
            batch_i += 1
            gen_output = gen.predict(x_imgs)
            d_loss_real = dis.train_on_batch([x_imgs, y_imgs], valid_label)
            d_loss_fake = dis.train_on_batch([x_imgs, gen_output], fake_label)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            g_loss = gan.train_on_batch(x_imgs, [valid_label, y_imgs]) 

            d_losses.append(d_loss[0])
            g_losses.append(g_loss[0])

            elapsed_time = datetime.datetime.now() - start_time

            print('',end='\r')
            print ("[Epoch %d/%d] [Batch %d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                        batch_i,
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss[0],
                                                                        elapsed_time), end='')
        #save weights
        if epoch % save_weights_each_epochs == 0:
            gen.save_weights(save_weights_path + 'gen.hdf5')
            gen.save_weights(save_weights_path + 'gen.backup.hdf5')
            dis.save_weights(save_weights_path + 'dis.hdf5')
            dis.save_weights(save_weights_path + 'dis.backup.hdf5')
            print('\nmodel saved')

            plt.clf()
            plt.plot(d_losses, label='discriminator')
            plt.plot(g_losses, label='generator')
            plt.legend(loc='best')
            plt.title('epoch ' + str(epoch))
            plt.savefig(save_weights_path + 'loss.png')

            _sample_test(gen, img_x_size, epoch, save_weights_path)
        
        if epoch % save_weights_checkpoint_each_epochs == 0:
            gen.save_weights(save_weights_path + 'gen%d.hdf5' % (epoch))
            dis.save_weights(save_weights_path + 'dis%d.hdf5' % (epoch))

def _sample_test(gen, img_x_size, epoch=0, save_sample_plot_path=''):
    x_imgs = load_sample_data(img_x_size)
    preds = []
    for x_img in x_imgs:
        pred = gen.predict(np.asarray([x_img]).reshape(1,img_x_size[0], img_x_size[1], 1))
        pred = pred * 0.5 + 0.5
        preds.append(pred[0].reshape(img_x_size[0], img_x_size[1]))
    
    fig, axs = plt.subplots(2, len(x_imgs))

    for i, pred in enumerate(preds):
        axs[0, i].imshow(x_imgs[i] * 0.5 + 0.5, cmap='gray')
        axs[1, i].imshow(pred, cmap='gray')
        axs[0, i].axis('off')
        axs[1, i].axis('off')
    
    fig.savefig(save_sample_plot_path + 'sample epoch %d.png' % (epoch,))
    plt.close()



if __name__ == '__main__':
    gan, gen, dis = make_gan(img_x_shape=(64, 64, 1), img_y_shape=(64, 64, 1),
        gen_dropout=0.2, dis_dropout=0.2)
    train(gan, gen, dis, img_x_size=(64, 64), img_y_size=(64, 64), epochs=5, batch_size=1)
