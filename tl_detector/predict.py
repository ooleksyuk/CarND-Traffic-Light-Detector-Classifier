import os
import numpy as np
import argparse

from skimage.io import imsave
from helper import print_text
from image_preprocessor import load_test_data
from model import pre_process, unet_model

MISS = 100
IMAGE_ROWS = 600
IMAGE_COLS = 800
PRED_DIR = 'preds'
MODEL_DIR = 'models'


def predict(parent_folder):
    print_text('Loading and pre-processing test data.')
    model = unet_model(parent_folder)

    test_images, test_image_names = load_test_data(parent_folder)
    test_images = pre_process(test_images)

    test_images = test_images.astype('float32')
    mean = np.mean(test_images)  # mean for data centering
    std = np.std(test_images)  # std for data normalization

    if parent_folder == "carla":
        test_images -= mean
        test_images /= std

    print_text('Loading saved weights.')
    model_json_name = 'tl_model_detector_' + str(parent_folder) + '.json'
    model_name = 'tl_weights_detector_' + str(parent_folder) + '.h5'
    print_text(model_json_name)
    model.load_weights(os.path.join(MODEL_DIR, model_name))

    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(MODEL_DIR, model_json_name), "w") as json_file:
        json_file.write(model_json)
    print_text('Saved model to disk')

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
