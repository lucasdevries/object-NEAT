import glob
from skimage import io , measure
import cv2
from scipy import ndimage
import numpy as np
from sklearn.model_selection import train_test_split

def sobel_op(image):
    dx = ndimage.sobel(image, 0)  # horizontal derivative
    dy = ndimage.sobel(image, 1)  # vertical derivative
    mag = np.hypot(dx, dy)  # magnitude
    mag = mag / np.max(mag)# normalize (Q&D)
    return mag

images_positive = []
labels_positive = []

images_negative = []
labels_negative = []

def load_data():
    for filename in glob.glob('positive_samples/*.bmp'): #assuming bmp
        image = io.imread(filename, as_grey=True)
        image = sobel_op(image)
        image = measure.block_reduce(image, (2,2), np.max)
        image = measure.block_reduce(image, (2,2), np.max)
        image = image.flatten()
        image = tuple(image)   
        images_positive.append(image)
        labels_positive.append((1.0,))

    for filename in glob.glob('negative_samples/*.bmp'): #assuming bmp
        image = io.imread(filename, as_grey=True)
        image = sobel_op(image)
        image = measure.block_reduce(image, (2,2), np.max)
        image = measure.block_reduce(image, (2,2), np.max)
        image = image.flatten()
        image = tuple(image)   
        images_negative.append(image)
        labels_negative.append((0.0,))

    X = images_positive[:400] + images_negative[:400]
    y = labels_positive[:400] + labels_negative[:400]

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify = y)
    return X_train, X_test, y_train, y_test