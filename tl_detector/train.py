import numpy as np
import argparse
import os

from skimage.io import imsave

from keras.callbacks import ModelCheckpoint, TensorBoard

from helper import print_text
from image_preprocessor import load_train_data, load_test_data
from model import pre_process, unet_model

PREDS_DIR = 'preds'
MODEL_DIR = 'models'


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

    tf_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    print_text('Creating and compiling model.')
    model = unet_model(parent_folder)
    file_model_name = 'tl_model_detector_' + str(parent_folder) + '.json'
    file_weights_name = 'tl_weights_detector_' + str(parent_folder) + '.h5'
    model_checkpoint = ModelCheckpoint(os.path.join(MODEL_DIR, file_weights_name), monitor='val_loss',
                                       save_best_only=True, save_weights_only=True, verbose=1)

    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(MODEL_DIR, file_model_name), "w") as json_file:
        json_file.write(model_json)
    print_text('Saved model to disk')

    print_text('Fitting model.')

    model.fit(imgs_train, imgs_mask_train, batch_size=16, epochs=100, verbose=1, shuffle=True,
              validation_split=0.2, callbacks=[model_checkpoint, tf_board])

    print_text('Loading and pre-processing test data.')
    imgs_test, test_image_names = load_test_data(parent_folder)
    imgs_test = pre_process(imgs_test)
    imgs_test = imgs_test.astype('float32')

    if parent_folder == "carla":
        imgs_test -= mean
        imgs_test /= std

    print_text('Loading saved weights.')
    model.load_weights(os.path.join(MODEL_DIR, file_weights_name))

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
