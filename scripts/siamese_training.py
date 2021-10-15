# -*- coding: utf-8 -*-

"""
Created on Fri Oct 15 13:36:38 2021

@author: aktas
"""

import tensorflow as tf
import math
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
import cv2
import glob
import os

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

MODEL_FNAME = 'embedding_network.h5'

def make_pairs(x, y):
    num_classes = max(y) + 1
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

    pairs = []
    labels = []

    for idx1 in range(len(x)):
        # add a matching example
        x1 = x[idx1]
        label1 = y[idx1]
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

CLASS_NAMES = ["covid", "normal", "pneumonia"]
classes = dict(zip(CLASS_NAMES, range(len(CLASS_NAMES))))

train_image_list = []
train_y_list = []
basedir = "dataset_siamese\\train\\"

for classdir in os.listdir(basedir):
    for filename in  os.listdir(basedir + classdir):
        impath = basedir +classdir + "\\" + filename
        if not os.path.isfile(impath):
            raise ValueError('Image name doesnt exist') 
        im = cv2.imread(impath, cv2.IMREAD_UNCHANGED)
        train_image_list.append(im)
        train_y_list.append(classes[classdir])
        
print("The train set contains",len(train_image_list)) 
print(train_y_list) 

valid_image_list = []
valid_y_list = []
basedir = "dataset_siamese\\validation\\"

for classdir in os.listdir(basedir):
    for filename in  os.listdir(basedir+classdir):
        impath = basedir + classdir + "\\" + filename
        if not os.path.isfile(impath):
            raise ValueError('Image name doesnt exist') 
        im = cv2.imread(impath, cv2.IMREAD_UNCHANGED)
        valid_image_list.append(im)
        valid_y_list.append(classes[classdir])
        
print("The valid set contains", len(valid_image_list))  
print(valid_y_list)

test_image_list = []
test_y_list = []
basedir = "dataset_siamese\\test\\"

for classdir in os.listdir(basedir):
    for filename in  os.listdir(basedir+classdir):
        impath = basedir + classdir + "\\" + filename
        if not os.path.isfile(impath):
            raise ValueError('Image name doesnt exist') 
        im = cv2.imread(impath, cv2.IMREAD_UNCHANGED)
        test_image_list.append(im)
        test_y_list.append(classes[classdir])
        
print("The test set contains", len(test_image_list))  
print(test_y_list)        

# make train pairs
pairs_train, labels_train = make_pairs(train_image_list, train_y_list)

# make validation pairs
pairs_val, labels_val = make_pairs(valid_image_list, valid_y_list)

# make test pairs
pairs_test, labels_test = make_pairs(test_image_list, test_y_list)

# """ L1 mistance - manhattan """
# def manhattan_distance(vects):
#    x, y = vects
#    return K.sum(K.abs(x-y),axis=1,keepdims=True)

# """ L2 distance """
# def euclidean_distance(vects):
#     x, y = vects
#     sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
#     return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


# def loss(margin=1):
#     def contrastive_loss(y_true, y_pred):
#         square_pred = tf.math.square(y_pred)
#         margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
#         return tf.math.reduce_mean((1 - y_true) * square_pred + (y_true) * margin_square)

#     return contrastive_loss


# # make train pairs
# pairs_train, labels_train = make_pairs(x_train, y_train)

# # make validation pairs
# pairs_val, labels_val = make_pairs(x_val, y_val)

# # make test pairs
# pairs_test, labels_test = make_pairs(x_test, y_test)


# input_1 = Input((100,100,3))
# input_2 = Input((100,100,3))


# embedding_network = tf.keras.models.load_model(MODEL_FNAME)

# # add here as the output of embedding network Towards the end of the pretrained model we add a flatten layer which is followed by a dense layer with 5120 neurons, sigmoid ac- tivation function, and L2 kernel regularizer

# tower_1 = embedding_network(input_1)
# tower_2 = embedding_network(input_2)

# merge_layer = manhattan_distance([tower_1, tower_2])
# output_layer = Dense(1, activation="sigmoid")(merge_layer)

# siamese = Model(inputs=[input_1, input_2], outputs=[output_layer])
# siamese.summary()

# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.000001)

# siamese.compile(loss=loss(1), optimizer="adam", metrics=["accuracy"])
# siamese.summary()
# history = siamese.fit([x_train_1, x_train_2],
#     labels_train,
#     validation_data=([x_val_1, x_val_2], labels_val),
#     batch_size=batch_size,
#     epochs=epochs,
#     callbacks = [early_stopping, checkpointer]
# )

