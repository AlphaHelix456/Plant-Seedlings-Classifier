import numpy as np
import pandas as pd
import cv2
import os
from keras.utils.np_utils import to_categorical

SPECIES = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen',
          'Loose Silky-bent', 'Maize','Scentless Mayweed', 'Shepherds Purse',
          'Small-flowered Cranesbill', 'Sugar beet']

SEED = 42


def load_train():
    train_data = []
    for species_id, sp in enumerate(SPECIES):
        for file in os.listdir(os.path.join('train', sp)):
            train_data.append(['train/{}/{}'.format(sp, file), species_id, sp])

    train = pd.DataFrame(train_data, columns=['Filepath', 'SpeciesId', 'Species'])

    # Randomize the order of training set
    train = train.sample(frac=1, random_state=SEED)
    train.index = np.arange(len(train))
    return train


def load_test():
    test_data = []
    for file in os.listdir('test'):
        test_data.append(['test/{}'.format(file), file])
    return pd.DataFrame(test_data, columns=['Filepath', 'File'])


def extract_labels(data):
    Y = data['SpeciesId'].values
    return to_categorical(Y, num_classes=12)

def extract_features(data, image_size):
    X = np.zeros(data.shape[0], image_size, image_size, 3)
    for i, file in enumerate(data['Filepath'].values):
        image = read_image(file)
        image_segmented = segment_image(image)
        X[i] = resize_image(image_segmented, (image_size, image_size))

    X = X / 255.
    return X


def read_image(filepath):
    return cv2.imread(filepath)


def resize_image(image, image_size):
    return cv2.resize(image.copy(), image_size, interpolation=cv2.INTER_AREA)


################  Image segmentation functions  ################


def create_mask(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Convert from BGR to HSV color-space to extract colored object
    lower_green = np.array([30, 100, 50])
    upper_green = np.array([85, 255, 255])
    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(image_hsv, lower_green, upper_green)
    # Uses a morphological operation called closing to close small holes in the image
    # We need a kernel or structuring element to determine the nature of the operation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def segment_image(image):
    mask = create_mask(image)
    # Bitwise-AND mask and original image
    return cv2.bitwise_and(image, image, mask=mask)
