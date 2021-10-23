# -*- coding: utf-8 -*-
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import cv2
import os

""" L1 mistance - manhattan """
def manhattan_distance(vects):
    x, y = vects
    return K.sum(K.abs(x-y), axis=1, keepdims=True)

""" L2 distance """
def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


def loss(margin=1):
    def contrastive_loss(y_true, y_pred):
        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean((1 - y_true) * square_pred + (y_true) * margin_square)

    return contrastive_loss

def make_pairs(x, y):
    num_classes = int(max(y) + 1)
    print("number of classes", num_classes)
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

    pairs = []
    labels = []

    for idx1 in range(len(x)):
        # add a matching example
        x1 = x[idx1]
        label1 = int(y[idx1])
        idx2 = random.choice(digit_indices[label1])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [1]

        # add a non-matching example
        label2 = random.randint(0, num_classes - 1)
        while label2 == label1:
            label2 = random.randint(0, num_classes - 1)

        idx2 = random.choice(digit_indices[label2])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [0]

    return np.array(pairs), np.array(labels).astype("float32")

    
""" Load the dataset and prepare pairs"""


def preprocess(array):
    """
    Normalize and resize the input array
    """
    processed_imgs = np.zeros((len(array), 100, 100, 3))
    
    for i, img in enumerate(array):
        # rescale to have values within 0 - 1 range [0,255] --> [0,1] 
        img = img.astype('float32') / 255.0
        
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # resize the image 
        img = cv2.resize(img, dsize=(100, 100))
        img = img[..., np.newaxis]
        # img = np.dstack((img, img))
        # img = np.dstack((img, img))
        # img = np.reshape(img, (100, 100, 3))
        
        # print(img.shape)
        # plt.imshow(img, cmap='gray')
        # plt.show()
        # return
        
        processed_imgs[i,:,:] = img
        
    return processed_imgs

# def load_images(basedir, path, inp):
#     image_list = []
#     y_list = []
    
#     CLASS_NAMES = ["covid", "normal", "pneumonia"]
#     classes = dict(zip(CLASS_NAMES, range(len(CLASS_NAMES))))

#     for classdir in os.listdir(os.path.join(basedir, path)):
#         for filename in os.listdir(os.path.join(basedir, path, classdir)):
#             impath = os.path.join(basedir, path, classdir, filename)
#             if not os.path.isfile(impath):
#                 raise ValueError('Image name doesnt exist') 
#             im = cv2.imread(impath, cv2.IMREAD_UNCHANGED)
#             image_list.append(im)
#             y_list.append(classes[classdir])
    
#     image_list = np.array(image_list, dtype=object)
#     y_list = np.array(y_list, dtype=object)
    
#     image_list = preprocess(image_list)   
#     print(type(image_list), type(y_list))     
#     print(image_list.shape, y_list.shape)    
#     return image_list, y_list

def load_images(basedir, path, input_size):
    batches = ImageDataGenerator(rescale = 1 / 255.).flow_from_directory(os.path.join(basedir, path),
                                                          target_size=(input_size),
                                                          batch_size=10000,
                                                          class_mode='binary',
                                                          shuffle=True,
                                                          seed=42) 

    return batches[0][0], batches[0][1]

def visualize(pairs, labels, to_show=6, num_col=3, predictions=None, test=False):
    """Creates a plot of pairs and labels, and prediction if it's test dataset.

    Arguments:
        pairs: Numpy Array, of pairs to visualize, having shape
               (Number of pairs, 2, 28, 28).
        to_show: Int, number of examples to visualize (default is 6)
                `to_show` must be an integral multiple of `num_col`.
                 Otherwise it will be trimmed if it is greater than num_col,
                 and incremented if if it is less then num_col.
        num_col: Int, number of images in one row - (default is 3)
                 For test and train respectively, it should not exceed 3 and 7.
        predictions: Numpy Array of predictions with shape (to_show, 1) -
                     (default is None)
                     Must be passed when test=True.
        test: Boolean telling whether the dataset being visualized is
              train dataset or test dataset - (default False).

    Returns:
        None.
    """

    # Define num_row
    # If to_show % num_col != 0
    #    trim to_show,
    #       to trim to_show limit num_row to the point where
    #       to_show % num_col == 0
    #
    # If to_show//num_col == 0
    #    then it means num_col is greater then to_show
    #    increment to_show
    #       to increment to_show set num_row to 1
    num_row = to_show // num_col if to_show // num_col != 0 else 1

    # `to_show` must be an integral multiple of `num_col`
    #  we found num_row and we have num_col
    #  to increment or decrement to_show
    #  to make it integral multiple of `num_col`
    #  simply set it equal to num_row * num_col
    to_show = num_row * num_col

    # Plot the images
    fig, axes = plt.subplots(num_row, num_col, figsize=(5, 5))
    for i in range(to_show):

        # If the number of rows is 1, the axes array is one-dimensional
        if num_row == 1:
            ax = axes[i % num_col]
        else:
            ax = axes[i // num_col, i % num_col]

        ax.imshow(tf.concat([pairs[i][0], pairs[i][1]], axis=1), cmap="gray")
        ax.set_axis_off()
        if test:
            ax.set_title("True: {} | Pred: {:.5f}".format(labels[i], predictions[i][0]))
        else:
            ax.set_title("Label: {}".format(labels[i]))
    if test:
        plt.tight_layout(rect=(0, 0, 1.9, 1.9), w_pad=0.0)
    else:
        plt.tight_layout(rect=(0, 0, 1.5, 1.5))
    plt.show()

def plt_metric(history, metric, title, has_valid=True):
    """Plots the given 'metric' from 'history'.

    Arguments:
        history: history attribute of History object returned from Model.fit.
        metric: Metric to plot, a string value present as key in 'history'.
        title: A string to be used as title of plot.
        has_valid: Boolean, true if valid data was passed to Model.fit else false.

    Returns:
        None.
    """
    plt.plot(history[metric])
    if has_valid:
        plt.plot(history["val_" + metric])
        plt.legend(["train", "validation"], loc="upper left")
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.show()



