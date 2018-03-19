import numpy as np
import os

from skimage.transform import resize
from skimage.color import rgb2grey

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K

IMAGE_ROWS = 96
IMAGE_COLS = 128
COLORS = 3
SMOOTH = 1.
ACTIVATION = 'relu'
PADDING = 'same'
KERNEL_SIZE = (3, 3)
STRIDES = (2, 2)
LEARN_RATE = 1e-5

MODEL_DIR = 'models'


def dice_coef(y_true, y_pred):
    return (2. * K.sum(K.flatten(y_true) * K.flatten(y_pred)) + SMOOTH) / (K.sum(y_true) + K.sum(y_pred) + SMOOTH)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def unet_model(parent_folder):
    inputs = Input((IMAGE_ROWS, IMAGE_COLS, 1))
    conv1 = Conv2D(filters=32, kernel_size=KERNEL_SIZE, activation=ACTIVATION, padding=PADDING, trainable=True)(inputs)
    conv1 = Conv2D(filters=32, kernel_size=KERNEL_SIZE, activation=ACTIVATION, padding=PADDING, name='conv_1_2',
                   trainable=True)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='maxpool_1', trainable=True)(conv1)

    conv2 = Conv2D(filters=64, kernel_size=KERNEL_SIZE, activation=ACTIVATION, padding=PADDING, name='conv_2_1',
                   trainable=True)(pool1)
    conv2 = Conv2D(filters=64, kernel_size=KERNEL_SIZE, activation=ACTIVATION, padding=PADDING, name='conv_2_2',
                   trainable=True)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='maxpool_2', trainable=True)(conv2)

    conv3 = Conv2D(filters=128, kernel_size=KERNEL_SIZE, activation=ACTIVATION, padding=PADDING, name='conv_3_1',
                   trainable=True)(pool2)
    conv3 = Conv2D(filters=128, kernel_size=KERNEL_SIZE, activation=ACTIVATION, padding=PADDING, name='conv_3_2',
                   trainable=True)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='maxpool_3', trainable=True)(conv3)

    conv4 = Conv2D(filters=256, kernel_size=KERNEL_SIZE, activation=ACTIVATION, padding=PADDING, name='conv_4_1',
                   trainable=True)(pool3)
    conv4 = Conv2D(filters=256, kernel_size=KERNEL_SIZE, activation=ACTIVATION, padding=PADDING, name='conv_4_2',
                   trainable=True)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), name='maxpool_4', trainable=True)(conv4)

    conv5 = Conv2D(filters=512, kernel_size=KERNEL_SIZE, activation=ACTIVATION, padding=PADDING, name='conv_5_1',
                   trainable=True)(pool4)
    conv5 = Conv2D(filters=512, kernel_size=KERNEL_SIZE, activation=ACTIVATION, padding=PADDING, name='conv_5_2',
                   trainable=True)(conv5)

    transpose_conv_5 = Conv2DTranspose(filters=256, kernel_size=(2, 2), strides=STRIDES, padding=PADDING,
                                       name='convtran_6', trainable=True)(conv5)
    up6 = concatenate([transpose_conv_5, conv4], name='up_6', trainable=True, axis=3)
    conv6 = Conv2D(filters=256, kernel_size=KERNEL_SIZE, activation=ACTIVATION, padding=PADDING, name='conv_6_1',
                   trainable=True)(up6)
    conv6 = Conv2D(filters=256, kernel_size=KERNEL_SIZE, activation=ACTIVATION, padding=PADDING, name='conv_6_2',
                   trainable=True)(conv6)

    transpose_conv_6 = Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=STRIDES, padding=PADDING,
                                       name='convtran_7', trainable=True)(conv6)
    up7 = concatenate([transpose_conv_6, conv3], name='up_7', trainable=True, axis=3)
    conv7 = Conv2D(filters=128, kernel_size=KERNEL_SIZE, activation=ACTIVATION, padding=PADDING, name='conv_7_1',
                   trainable=True)(up7)
    conv7 = Conv2D(filters=128, kernel_size=KERNEL_SIZE, activation=ACTIVATION, padding=PADDING, name='conv_7_2',
                   trainable=True)(conv7)

    transpose_conv_7 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=STRIDES, padding=PADDING,
                                       name='convtran_8', trainable=True)(conv7)
    up8 = concatenate([transpose_conv_7, conv2], name='up_8', trainable=True, axis=3)
    conv8 = Conv2D(filters=64, kernel_size=KERNEL_SIZE, activation=ACTIVATION, padding=PADDING, name='conv_8_1',
                   trainable=True)(up8)
    conv8 = Conv2D(filters=64, kernel_size=KERNEL_SIZE, activation=ACTIVATION, padding=PADDING, name='conv_8_2',
                   trainable=True)(conv8)

    transpose_conv_8 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=STRIDES, padding=PADDING,
                                       name='convtran_9', trainable=True)(conv8)
    up9 = concatenate([transpose_conv_8, conv1], name='up_9', trainable=True, axis=3)
    conv9 = Conv2D(filters=32, kernel_size=KERNEL_SIZE, activation=ACTIVATION, padding=PADDING, name='conv_9_1',
                   trainable=True)(up9)
    conv9 = Conv2D(filters=32, kernel_size=KERNEL_SIZE, activation=ACTIVATION, padding=PADDING, name='conv_9_2',
                   trainable=True)(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    model.load_weights(os.path.join(MODEL_DIR, 'weights.h5'), by_name=True)
    model.compile(optimizer=Adam(lr=LEARN_RATE), loss=dice_coef_loss, metrics=[dice_coef])

    model.summary()

    return model


def pre_process(images):
    images_p = np.ndarray((images.shape[0], IMAGE_ROWS, IMAGE_COLS), dtype=np.uint8)
    for i in range(images.shape[0]):
        images_p[i] = rgb2grey(resize(images[i], (IMAGE_ROWS, IMAGE_COLS, COLORS),
                                      preserve_range=True, mode="constant"))

    images_p = images_p[..., np.newaxis]
    return images_p
