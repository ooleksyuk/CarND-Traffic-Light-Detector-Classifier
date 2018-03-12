import numpy as np
import argparse
import os

from skimage.io import imsave

from keras.callbacks import ModelCheckpoint

from helper import print_text
from image_preprocessor import load_train_data, load_test_data
from model import pre_process, get_unet

PREDS_DIR = 'preds'


def train_and_predict(parent_folder):
    print_text('Loading and pre-processing train data.')
    imgs_train, imgs_mask_train = load_train_data(parent_folder)
    imgs_train = pre_process(imgs_train)
    imgs_mask_train = pre_process(imgs_mask_train)
    imgs_train = imgs_train.astype('float32')

    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    if parent_folder == "carla":
        imgs_train -= mean
        imgs_train /= std
    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    print_text('Creating and compiling model.')
    model = get_unet(parent_folder)
    model_checkpoint = ModelCheckpoint('tl_weights.h5', monitor='val_loss', save_best_only=True)

    print_text('Fitting model.')

    model.fit(imgs_train, imgs_mask_train, batch_size=16, epochs=30, verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint])

    print_text('Loading and pre-processing test data.')
    imgs_test, test_image_names = load_test_data(parent_folder)
    imgs_test = pre_process(imgs_test)
    imgs_test = imgs_test.astype('float32')

    if parent_folder == "carla":
        imgs_test -= mean
        imgs_test /= std

    print_text('Loading saved weights.')
    model.load_weights(os.path.join(parent_folder, 'tl_weights.h5'))

    print_text('Predicting masks on test data.')
    predicted_image_masks = model.predict(imgs_test, verbose=1)

    print_text('Saving predicted masks to files.')
    if not os.path.exists(PREDS_DIR):
        os.mkdir(PREDS_DIR)
    for image_mask, image_name in zip(predicted_image_masks, test_image_names):
        image_mask = (image_mask[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(PREDS_DIR, image_name + '.pred.png'), image_mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'parent_folder',
        type=str,
        nargs='?',
        default='carla',
        help='Path folder with mask images.'
    )

    args = parser.parse_args()
    train_and_predict(args.parent_folder)
