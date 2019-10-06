import tensorflow as tf
from keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, Dropout, Conv2DTranspose, Activation
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
from keras.initializers import RandomNormal
from PIL import Image
rcParams['figure.figsize'] = 14, 8

def make_discriminator(img_x_shape, img_y_shape, dropout=0, init_filters_n=64,
    use_label=False, label_embed_size=50, label_classes_n=None, predict_class=False,
    filter_size=4, use_binary_validity=False, binary_validity_dropout_rate=0):

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

    d1 = d_layer(d0, init_filters_n, f_size=filter_size, bn = False, dropout_rate=dropout)
    d2 = d_layer(d1, init_filters_n * 2, f_size=filter_size,  dropout_rate=dropout)
    d3 = d_layer(d2, init_filters_n * 4, f_size=filter_size, dropout_rate=dropout)
    d4 = d_layer(d3, init_filters_n * 8, f_size=filter_size, dropout_rate=dropout)

    if not use_binary_validity:
        validity = Conv2D(1, kernel_size=4, activation='sigmoid', strides=1, padding='same')(d4)
    else:
        flatten_layer = Flatten()(d4)
        validity = Dense(1, activation='sigmoid', name='dis_valid_binary')(flatten_layer)

    if predict_class:
        ly = Flatten()(d4)
        if binary_validity_dropout_rate:
            ly = Dropout(binary_validity_dropout_rate)(ly)
        predict_class_layer = Dense(label_classes_n, activation='softmax', name='dense_classes')(ly)
        
        dis_output = [validity, predict_class_layer]
    else:
        dis_output = validity


    if use_label:
        model = Model([img_X, img_Y, label_input_layer], dis_output)
    else:
        model =  Model([img_X, img_Y], dis_output)
    return model

def make_discriminator2(img_x_shape, img_y_shape, init_filters_n=64):
    def conv2d(layer_in, filters, f_size=4, bn=True):
        ly = Conv2D(filters, f_size, strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
        if bn:
            ly = BatchNormalization()(ly)
        ly = LeakyReLU(alpha=0.2)(ly)
        return ly

    #input layer
    init = RandomNormal(stddev=0.02)
    input_x_layer = Input(shape=img_x_shape)
    input_y_layer = Input(shape=img_y_shape)
    input_merged_layer = Concatenate()([input_x_layer, input_y_layer])

    #conv
    ly = conv2d(input_merged_layer, init_filters_n, bn=False)
    ly = conv2d(ly, init_filters_n * 2)
    ly = conv2d(ly, init_filters_n * 4)
    ly = conv2d(ly, init_filters_n * 8)

    #second last layer
    ly = Conv2D(init_filters_n * 8, (4,4), padding='same', kernel_initializer=init)(ly)
    ly = BatchNormalization()(ly)
    ly = LeakyReLU(alpha=0.2)(ly)

    #patch output
    ly = Conv2D(1, (4,4), activation='sigmoid', padding='same', kernel_initializer=init)(ly)

    return Model([input_x_layer, input_y_layer], ly)

def make_generator2(img_shape, init_filters_n=64, dropout_rate=None):
    init = RandomNormal(stddev=0.02)

    def encoder(layer_in, filters, f_size=4, bn=True):
        ly = Conv2D(filters, f_size, strides = (2,2), padding='same', kernel_initializer=init)(layer_in)
        if bn:
            ly = BatchNormalization()(ly)
        ly = LeakyReLU(alpha=0.2)(ly)
        return ly
    
    def decoder(layer_in, skip_in, filters, f_size=4, dropout_rate=None):
        ly = Conv2DTranspose(filters, f_size, strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
        ly = BatchNormalization()(ly)
        if dropout_rate:
            ly = Dropout(dropout_rate)(ly)
        ly = Concatenate()([ly, skip_in]) 
        ly = Activation('relu')(ly)
        return ly
    
    #encoder
    input_layer = Input(shape=img_shape)
    e1 = encoder(input_layer, init_filters_n, bn=False) 
    e2 = encoder(e1, init_filters_n * 2)
    e3 = encoder(e2, init_filters_n * 4)
    e4 = encoder(e3, init_filters_n * 8)

    #bottleneck
    b = Conv2D(init_filters_n * 8, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e4)
    b = Activation('relu')(b)

    #decoder
    d1 = decoder(b, e4, init_filters_n * 8)
    d2 = decoder(d1, e3, init_filters_n *4)
    d3 = decoder(d2, e2, init_filters_n *2, dropout_rate=dropout_rate)
    d4 = decoder(d3, e1, init_filters_n, dropout_rate=dropout_rate)

    #output
    o = Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d4)
    o = Activation('tanh')(o)        

    return Model(input_layer, o)

def make_generator(img_y_shape, dropout=0, init_filters_n=64, channels=1,
    use_label=False, label_embed_size=50, label_classes_n=None, filter_size=4):
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
    d1 = conv2d(d0, init_filters_n, f_size=filter_size, bn=False, dropout_rate=dropout)
    d2 = conv2d(d1, init_filters_n*2, f_size=filter_size, dropout_rate=dropout)
    d3 = conv2d(d2, init_filters_n*4, f_size=filter_size, dropout_rate=dropout)
    d4 = conv2d(d3, init_filters_n*8, f_size=filter_size, dropout_rate=dropout)

    # Upsampling
    u1 = deconv2d(d4, d3, init_filters_n*4, f_size=filter_size, dropout_rate=dropout)
    u2 = deconv2d(u1, d2, init_filters_n*2, f_size=filter_size, dropout_rate=dropout)
    u3 = deconv2d(u2, d1, init_filters_n  , f_size=filter_size, dropout_rate=dropout)

    u7 = UpSampling2D(size=2)(u3)
    output_img = Conv2D(channels, kernel_size=filter_size, strides=1, padding='same', activation='tanh')(u7)

    if use_label:
        model = Model([image_input_layer, label_input_layer], output_img)
    else:
        model = Model(d0, output_img)
    return model

def make_gan(img_x_shape, img_y_shape, init_filters_n=64, dis_dropout=0, gen_dropout=0,
    use_label=False, label_embed_size=50, label_classes_n=None, predict_class=False,
    filter_size=4, use_binary_validity=False, binary_validity_dropout_rate=0,
    gan_loss_weights=None, use_generator2=False, use_discriminator2=False,
    gen_init_filters_n=None, dis_init_filters_n=None):
    optimizer = Adam(0.0002, 0.5)

    #discriminator
    dis_filters = dis_init_filters_n if dis_init_filters_n is not None else init_filters_n
    if not use_discriminator2:
        dis = make_discriminator(img_x_shape, img_y_shape, dis_dropout, init_filters_n=dis_filters,
            use_label=use_label, label_embed_size=label_embed_size,
            label_classes_n=label_classes_n, predict_class=predict_class,
            use_binary_validity=use_binary_validity, binary_validity_dropout_rate=binary_validity_dropout_rate)
    else:
        dis = make_discriminator2(img_x_shape, img_y_shape, init_filters_n=dis_filters)
    
    dis_validity_loss = 'binary_crossentropy'

    if predict_class:
        dis.compile(loss=[dis_validity_loss, 'sparse_categorical_crossentropy'],
            optimizer = optimizer,
            metrics=['accuracy'], loss_weights=[0.5])
    else:
        dis.compile(loss=dis_validity_loss, optimizer=optimizer, metrics=['accuracy'], loss_weights=[0.5])
    
    #generator
    gen_filters = gen_init_filters_n if gen_init_filters_n is not None else init_filters_n
    if not use_generator2:
        gen = make_generator(img_y_shape, gen_dropout, init_filters_n = gen_filters,
            use_label=use_label, label_embed_size=label_embed_size,
            label_classes_n=label_classes_n, filter_size=filter_size)
    else:
        gen = make_generator2(img_y_shape, init_filters_n=gen_filters, dropout_rate=gen_dropout)

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
        gan_loss = [dis_validity_loss, 'sparse_categorical_crossentropy', 'mae']
        if gan_loss_weights is None:
            gan_loss_weights = [1, 1, 100]
    else:
        validity_layer = dis_output_layer
        gan_output_layer = [validity_layer, gen_output_layer]
        gan_loss = [dis_validity_loss, 'mae']
        if gan_loss_weights is None:
            gan_loss_weights = [1, 100]

    gan = Model(inputs=x_input_layer, outputs=gan_output_layer)
    gan.compile(loss=gan_loss,
        loss_weights=gan_loss_weights,
        optimizer=optimizer)
    
    return gan, gen, dis

def train(dataset_cache, gan, gen, dis, img_x_size, img_y_size, epochs, batch_size,
    use_label=False, predict_class=False, use_binary_validity=False, invert_color=False,
    label_classes_n=44, augment=True,
    init_epoch=1,
    save_weights_each_epochs=1,
    save_weights_checkpoint_each_epochs=5,
    save_weights_path=''):
    start_time = datetime.datetime.now()

    #make patch label
    patch = int(img_x_size[1] / 2**4)
    disc_patch = (patch, patch, 1)
    valid_label = np.ones((batch_size, 1)) if use_binary_validity else np.ones((batch_size,) + disc_patch)
    fake_label = np.zeros((batch_size, 1)) if use_binary_validity else np.zeros((batch_size,) + disc_patch)

    d_loss_avgs = []
    g_loss_avgs = []

    for epoch in range(init_epoch, epochs+1):
        d_losses = []
        g_losses = []
        batch_i = 0
        for chrunk in make_dataset_generator(batch_size, dataset_cache, img_x_size, img_y_size, use_label=use_label, invert_color=invert_color, augment=augment):
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

            #train discriminator
            if not use_label:
                gen_output = gen.predict(x_imgs)
                d_loss_real = dis.train_on_batch([x_imgs, y_imgs], dis_y_valid)
                d_loss_fake = dis.train_on_batch([x_imgs, gen_output], dis_y_fake)
            elif use_label:
                gen_output = gen.predict([x_imgs, class_labels])
                d_loss_real = dis.train_on_batch([x_imgs, y_imgs, class_labels], dis_y_valid)
                d_loss_fake = dis.train_on_batch([x_imgs, gen_output, class_labels], dis_y_fake)

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            #train generator
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

        #end of epoch
        d_loss_avg = np.asarray(d_losses).mean()
        g_loss_avg = np.asarray(g_losses).mean()
        d_loss_avgs.append(d_loss_avg)
        g_loss_avgs.append(g_loss_avg)
        #plot loss
        plt.clf()
        plt.plot(d_loss_avgs, label='discriminator')
        plt.plot(g_loss_avgs, label='generator')
        plt.legend(loc='best')
        plt.title('epoch ' + str(epoch))
        plt.savefig(save_weights_path + 'loss.png')

        #save weights
        if epoch % save_weights_each_epochs == 0:
            gen.save_weights(save_weights_path + 'gen.hdf5')
            gen.save_weights(save_weights_path + 'gen.backup.hdf5')
            dis.save_weights(save_weights_path + 'dis.hdf5')
            dis.save_weights(save_weights_path + 'dis.backup.hdf5')
            print('\nmodel saved')

            _sample_test(gen, img_x_size, epoch, save_weights_path, use_label=use_label, invert_color=invert_color)
        
        if epoch % save_weights_checkpoint_each_epochs == 0:
            gen.save_weights(save_weights_path + 'gen%d.hdf5' % (epoch))
            dis.save_weights(save_weights_path + 'dis%d.hdf5' % (epoch))

def _sample_test(gen, img_x_size, epoch=0, save_sample_plot_path='', use_label=False, invert_color=False):
    x_imgs, labels = load_sample_data(img_x_size, invert_color=invert_color)
    preds = []
    for x_img, label in zip(x_imgs, labels):
        reshaped = np.asarray([x_img]).reshape(1,img_x_size[0], img_x_size[1], 1)
        if use_label:
            pred = gen.predict([reshaped, np.asarray([label])])
        elif not use_label:
            pred = gen.predict(reshaped)
        pred = 1 - (pred * 0.5 + 0.5) if invert_color else (pred * 0.5 - 0.5)
        preds.append(pred[0].reshape(img_x_size[0], img_x_size[1]))
    
    fig, axs = plt.subplots(2, len(x_imgs))

    for i, pred in enumerate(preds):
        img_to_show = 1 - (x_imgs[i] * 0.5 + 0.5) if invert_color else x_imgs[i] * 0.5 + 0.5
        axs[0, i].imshow(img_to_show, cmap='gray')
        axs[1, i].imshow(pred, cmap='gray')
        axs[0, i].axis('off')
        axs[1, i].axis('off')
    
    fig.savefig(save_sample_plot_path + 'sample epoch %d.png' % (epoch,))
    plt.close()

def predict(gen, img_size, x_path, y_path, invert_color=False):
    #load x files
    x_imgs = []
    for fn in os.listdir(x_path):
        img = read_img(x_path + fn, img_size)
        reshaped = np.asarray([img]).reshape(1, img_size[0], img_size[1], 1)
        pred = gen.predict(reshaped)
        pred = 1-(pred * 0.5 + 0.5) if invert_color else (pred * 0.5 + 0.5)
        
        # save to y_path
        img = pred[0].reshape( img_size[0], img_size[1])
        img = Image.fromarray(np.uint(img * 255))
        save_fn = fn[:-4] + '.bmp'
        img.convert('RGB').save(y_path + save_fn)

if __name__ == '__main__':
    dataset_cache = make_dataset_cache((128, 128), (128, 128))

    gan, gen, dis = make_gan(img_x_shape=(128, 128, 1), img_y_shape=(128, 128, 1), init_filters_n = 64, filter_size=4,
      use_generator2=True, use_discriminator2=True,
      gen_dropout=0.5, dis_dropout=0, gan_loss_weights=[1, 100])

    # gen.load_weights('gen.hdf5')
    # dis.load_weights('dis.hdf5')

    train(dataset_cache, gan, gen, dis, img_x_size=(128, 128), img_y_size=(128, 128), init_epoch=1, augment=False,
        epochs=9999, batch_size=1, save_weights_each_epochs=1, save_weights_checkpoint_each_epochs=9999)


    # predict(gen, img_size=(128,128), x_path='x_path/', y_path='y_path/')


        