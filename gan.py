import tensorflow as tf
from keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, Dropout
from keras.layers import Concatenate, Embedding, Dense, Reshape, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
from dataset_util import make_dataset_generator, load_sample_data, read_img, make_dataset_cache
import datetime
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
from keras.utils import to_categorical
from PIL import Image
rcParams['figure.figsize'] = 14, 8

def make_discriminator(img_x_shape, img_y_shape, dropout=0, init_filters_n=64,
    use_label=False, label_embed_size=50, label_classes_n=None, predict_class=False):

    def d_layer(layer_input, filters, f_size=4, bn=True, dropout_rate=0):
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        if dropout_rate:
            d = Dropout(dropout_rate)(d)
        return d

    #input layer
    img_X = Input(shape=img_x_shape)
    img_Y = Input(shape=img_y_shape)
    if use_label:
        label_input_layer = Input(shape=(1,))
        embed_layer = Embedding(label_classes_n, label_embed_size)(label_input_layer)
        embed_dense_1d = Dense(img_y_shape[0] * img_y_shape[1])(embed_layer)
        embed_dense_2d = Reshape((img_y_shape[0], img_y_shape[1], 1))(embed_dense_1d)
        d0 = Concatenate()([img_X, img_Y, embed_dense_2d])
    elif not use_label:
        d0 = Concatenate()([img_X, img_Y])

    d1 = d_layer(d0, init_filters_n, bn = False, dropout_rate=dropout)
    d2 = d_layer(d1, init_filters_n * 2, dropout_rate=dropout)
    d3 = d_layer(d2, init_filters_n * 4, dropout_rate=dropout)
    d4 = d_layer(d3, init_filters_n * 8, dropout_rate=dropout)

    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

    if predict_class:
        flatten_layer = Flatten()(d4)
        predict_class_layer = Dense(label_classes_n, activation='softmax', name='dense_classes')(flatten_layer)
        dis_output = [validity, predict_class_layer]
    else:
        dis_output = validity


    if use_label:
        model = Model([img_X, img_Y, label_input_layer], dis_output)
    else:
        model =  Model([img_X, img_Y], dis_output)
    return model

def make_generator(img_y_shape, dropout=0, init_filters_n=64, channels=1,
    use_label=False, label_embed_size=50, label_classes_n=None):
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
    if use_label:
        label_input_layer = Input(shape=(1,))
        embed_layer = Embedding(label_classes_n, label_embed_size)(label_input_layer)
        embed_dense_1d = Dense(img_y_shape[0] * img_y_shape[1])(embed_layer)
        embed_dense_2d = Reshape((img_y_shape[0], img_y_shape[1], 1))(embed_dense_1d)
        image_input_layer = Input(shape=img_y_shape)
        d0 = Concatenate()([image_input_layer, embed_dense_2d])
    elif not use_label:
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

    if use_label:
        model = Model([image_input_layer, label_input_layer], output_img)
    else:
        model = Model(d0, output_img)
    return model

def make_gan(img_x_shape, img_y_shape, dis_dropout, gen_dropout,
    use_label=False, label_embed_size=50, label_classes_n=None, predict_class=False):
    optimizer = Adam(0.0002, 0.5)

    dis = make_discriminator(img_x_shape, img_y_shape, dis_dropout,
        use_label=use_label, label_embed_size=label_embed_size,
        label_classes_n=label_classes_n, predict_class=predict_class)
    
    if predict_class:
        dis.compile(loss=['mse', 'sparse_categorical_crossentropy'],
            optimizer = optimizer,
            metrics=['accuracy'])
    else:
        dis.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
    
    gen = make_generator(img_y_shape, gen_dropout,
        use_label=use_label, label_embed_size=label_embed_size,
        label_classes_n=label_classes_n)

    image_input_layer = Input(shape=img_x_shape)
    if use_label:
        label_input_layer = Input(shape=(1,))
        x_input_layer = [image_input_layer,label_input_layer]
    else:
        x_input_layer = image_input_layer
    
    gen_output_layer = gen(x_input_layer)

    dis.trainable = False

    if use_label:
        dis_output_layer = dis([image_input_layer, gen_output_layer, label_input_layer])
    else:
        dis_output_layer = dis([image_input_layer, gen_output_layer])
    
    if predict_class:
        validity_layer, pred_class_layer = dis_output_layer
        gan_output_layer = [validity_layer, pred_class_layer, gen_output_layer]
        gan_loss = ['mse', 'sparse_categorical_crossentropy', 'mae']
        gan_loss_weights = [1, 1, 100]
    else:
        validity_layer = dis_output_layer
        gan_output_layer = [validity_layer, gen_output_layer]
        gan_loss = ['mse', 'mae']
        gan_loss_weights = [1, 100]

    gan = Model(inputs=x_input_layer, outputs=gan_output_layer)
    gan.compile(loss=gan_loss,
        loss_weights=gan_loss_weights,
        optimizer=optimizer)
    
    return gan, gen, dis

def train(dataset_cache, gan, gen, dis, img_x_size, img_y_size, epochs, batch_size,
    use_label=False, predict_class=False,
    init_epoch=1,
    label_classes_n=44,
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
        for chrunk in make_dataset_generator(batch_size, dataset_cache, img_x_size, img_y_size, use_label=use_label):
            x_imgs = chrunk[0] 
            y_imgs = chrunk[1]
            if use_label:
                class_labels = chrunk[2]
            batch_i += 1

            if predict_class:
                dis_y_valid = [valid_label, class_labels]
                dis_y_fake = [fake_label, class_labels]
            else:
                dis_y_valid = valid_label
                dis_y_fake = fake_label

            if not use_label:
                gen_output = gen.predict(x_imgs)
                d_loss_real = dis.train_on_batch([x_imgs, y_imgs], dis_y_valid)
                d_loss_fake = dis.train_on_batch([x_imgs, gen_output], dis_y_fake)
            elif use_label:
                gen_output = gen.predict([x_imgs, class_labels])
                d_loss_real = dis.train_on_batch([x_imgs, y_imgs, class_labels], dis_y_valid)
                d_loss_fake = dis.train_on_batch([x_imgs, gen_output, class_labels], dis_y_fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            if not use_label:
                g_loss = gan.train_on_batch(x_imgs, [valid_label, class_labels, y_imgs] if predict_class else [valid_label, y_imgs])
            elif use_label:
                g_loss = gan.train_on_batch([x_imgs, class_labels], [valid_label, class_labels, y_imgs] if predict_class else [valid_label, y_imgs])

            d_losses.append(d_loss[0])
            g_losses.append(g_loss[0])

            elapsed_time = datetime.datetime.now() - start_time

            print('',end='\r')

            if predict_class:
                print ("[Epoch %d/%d] [Batch %d] [D loss: %f, judge_acc: %3d%%, classify_acc: %3d%% ] [G loss: %f] time: %s" % (epoch, epochs,
                                                                        batch_i,
                                                                        d_loss[0], 100*d_loss[3], 100*d_loss[4],
                                                                        g_loss[0],
                                                                        elapsed_time), end='')
            else:
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

            _sample_test(gen, img_x_size, epoch, save_weights_path, use_label=use_label)
        
        if epoch % save_weights_checkpoint_each_epochs == 0:
            gen.save_weights(save_weights_path + 'gen%d.hdf5' % (epoch))
            dis.save_weights(save_weights_path + 'dis%d.hdf5' % (epoch))

def _sample_test(gen, img_x_size, epoch=0, save_sample_plot_path='', use_label=False):
    x_imgs, labels = load_sample_data(img_x_size)
    preds = []
    for x_img, label in zip(x_imgs, labels):
        reshaped = np.asarray([x_img]).reshape(1,img_x_size[0], img_x_size[1], 1)
        if use_label:
            pred = gen.predict([reshaped, np.asarray([label])])
        elif not use_label:
            pred = gen.predict(reshaped)
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

def predict(gen, img_size, x_path, y_path):
    #load x files
    x_imgs = []
    for fn in os.listdir(x_path):
        img = read_img(x_path + fn, img_size)
        reshaped = np.asarray([img]).reshape(1, img_size[0], img_size[1], 1)
        pred = gen.predict(reshaped)
        pred = pred * 0.5 + 0.5
        
        # save to y_path
        img = pred[0].reshape( img_size[0], img_size[1])
        img = Image.fromarray(np.uint(img * 255))
        save_fn = fn[:-4] + '.bmp'
        img.convert('RGB').save(y_path + save_fn)

if __name__ == '__main__':
    dataset_cache = make_dataset_cache((64, 64), (64, 64))
    gan, gen, dis = make_gan(img_x_shape=(64, 64, 1), img_y_shape=(64, 64, 1),
        # use_label=True, label_classes_n=44,
        gen_dropout=0.2, dis_dropout=0.2)
    train(dataset_cache, gan, gen, dis, img_x_size=(64, 64), img_y_size=(64, 64), 
        # use_label=True,
        epochs=5, batch_size=1)

    # gan, gen, dis = make_gan(img_x_shape=(128, 128, 1), img_y_shape=(128,128,1),
    # dis_dropout=0.2, gen_dropout=0.2, label_classes_n=44)
    # gen.load_weights('gen.hdf5')

    # predict(gen, img_size=(128,128), x_path='x_path/', y_path='y_path/')


        