import os
import numpy as np
import argparse

from skimage.io import imsave
from helper import print_text
from image_preprocessor import load_test_data
from model import pre_process, get_unet

MISS = 100
IMAGE_ROWS = 600
IMAGE_COLS = 800
PRED_DIR = 'preds'


def predict(parent_folder):
    print_text('Loading and pre-processing test data.')
    model = get_unet(parent_folder)

    test_images, test_image_names = load_test_data(parent_folder)
    test_images = pre_process(test_images)

    test_images = test_images.astype('float32')
    mean = np.mean(test_images)  # mean for data centering
    std = np.std(test_images)  # std for data normalization

    if parent_folder == "carla":
        test_images -= mean
        test_images /= std

    print_text('Loading saved weights.')
    if parent_folder == 'carla':
        model_name = 'tl_detector_carla.h5'
    else:
        model_name = 'tl_detector_sim.h5'
    model.load_weights(model_name)

    print_text('Predicting masks on test data.')
    predicted_image_masks = model.predict(test_images, verbose=1)

    print_text('Saving predicted masks to files.')
    if not os.path.exists(PRED_DIR):
        os.mkdir(PRED_DIR)
    for pred_image, image_name in zip(predicted_image_masks, test_image_names):
        pred_image = (pred_image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(PRED_DIR, image_name + '.pred.png'), pred_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'folder',  # carla or sim
        type=str,
        help='Path folder with images to predict.'
    )
    args = parser.parse_args()

    predict(args.folder)
