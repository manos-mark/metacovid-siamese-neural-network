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

    return np.array(pairs, dtype=object), np.array(labels, dtype=object).astype("float32")

    
""" Load the dataset and prepare pairs"""


def preprocess(array):
    """
    Normalize and resize the input array
    """
    processed_imgs = np.zeros((len(array), 100, 100, 3))
    
    print(array[0].shape)
    for i, img in enumerate(array):
        # rescale to have values within 0 - 1 range [0,255] --> [0,1] 
        img = img.astype('float32') / 255.0
        
        # resize the image 
        img = np.resize(img, (100, 100, 3))
        
        processed_imgs[i,:,:] = img
        
    return processed_imgs

def load_images(base_dir, path):
    image_list = []
    y_list = []

    for classdir in os.listdir(os.path.join(basedir, path)):
        for filename in os.listdir(os.path.join(basedir, path, classdir)):
            impath = os.path.join(basedir, path, classdir, filename)
            if not os.path.isfile(impath):
                raise ValueError('Image name doesnt exist') 
            im = cv2.imread(impath, cv2.IMREAD_UNCHANGED)
            image_list.append(im)
            y_list.append(classes[classdir])
    
    image_list = np.array(image_list, dtype=object)
    y_list = np.array(y_list, dtype=object)
    
    image_list = preprocess(image_list)        
    return image_list, y_list
        
CLASS_NAMES = ["covid", "normal", "pneumonia"]
classes = dict(zip(CLASS_NAMES, range(len(CLASS_NAMES))))


basedir = os.path.join("dataset", "siamese") 

train_image_list, train_y_list = load_images(basedir, 'train')
print("The train set contains",len(train_image_list)) 

valid_image_list, valid_y_list = load_images(basedir, 'validation')   
print("The valid set contains", len(valid_image_list))  

# test_image_list, test_y_list = load_images(basedir, 'test')   
# print("The test set contains", len(test_image_list))  


# make train pairs
pairs_train, labels_train = make_pairs(train_image_list, train_y_list)

# make validation pairs
pairs_val, labels_val = make_pairs(valid_image_list, valid_y_list)

# make test pairs
#pairs_test, labels_test = make_pairs(test_image_list, test_y_list)

x_train_1 = pairs_train[:, 0]  
x_train_2 = pairs_train[:, 1]

print(type(x_train_1))# np.array
print("number of pairs for training", np.shape(x_train_1)[0]) # we have 60 pairs

print(np.shape(x_train_1[0]))

x_val_1 = pairs_val[:, 0] 
x_val_2 = pairs_val[:, 1]

print(type(x_train_1))# np.array
print("number of pairs for validation", np.shape(x_train_1)[0]) # we have 60 pairs

# x_test_1 = pairs_test[:, 0] 
# x_test_2 = pairs_test[:, 1]

print(type(x_train_1))# np.array
print("number of pairs for test", np.shape(x_train_1)[0]) # we have 60 pairs

""" L1 mistance - manhattan """
def manhattan_distance(vects):
    x, y = vects
    return K.sum(K.abs(x-y),axis=1,keepdims=True)

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


input_1 = Input((100,100,3))
input_2 = Input((100,100,3))


embedding_network = tf.keras.models.load_model(MODEL_FNAME)


# add here as the output of embedding network Towards the end of the pretrained model we add a flatten layer which is followed by a dense layer with 5120 neurons, sigmoid ac- tivation function, and L2 kernel regularizer

tower_1 = embedding_network(input_1)
tower_2 = embedding_network(input_2)

merge_layer = manhattan_distance([tower_1, tower_2])
output_layer = Dense(1, activation="sigmoid")(merge_layer)

siamese = Model(inputs=[input_1, input_2], outputs=[output_layer])
siamese.summary()

""" callbacks """

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.000001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
checkpointer = ModelCheckpoint(filepath='siamese_network.h5', verbose=1, 
                                save_best_only=True)


x_train_1_tensor = tf.convert_to_tensor(x_train_1, dtype=tf.float32)
x_train_2_tensor = tf.convert_to_tensor(x_train_2, dtype=tf.float32)

x_val_1_tensor = tf.convert_to_tensor(x_val_1, dtype=tf.float32)
x_val_2_tensor = tf.convert_to_tensor(x_val_2, dtype=tf.float32)

""" train the model """

siamese.compile(loss=loss(1), optimizer="adam", metrics=["accuracy"])
siamese.summary()
history = siamese.fit([x_train_1_tensor, x_train_2_tensor],
    labels_train,
    validation_data=([x_val_1_tensor, x_val_2_tensor], labels_val),
    batch_size=1,
    epochs=175,   # 175 for contrastive 100 for cross ent
    callbacks = [early_stopping, reduce_lr]
)

