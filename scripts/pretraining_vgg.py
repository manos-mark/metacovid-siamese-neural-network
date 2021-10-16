# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math

import scripts.utils as utils

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import vgg16, imagenet_utils
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import binary_crossentropy
from tensorflow.keras.layers import Dense, Flatten, Input, Lambda, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
import itertools

from skimage import exposure

import logging
from tensorflow.keras.models import load_model

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

print('Using:')
print('\t\u2022 TensorFlow version:', tf.__version__)
print('\t\u2022 tf.keras version:', tf.keras.__version__)
print('\t\u2022 Running on GPU' if tf.test.is_gpu_available() else '\t\u2022 GPU device not found. Running on CPU')

""" load the datasets """

base_dir = os.path.join('dataset', 'pretrain')

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

train_covid_dir = os.path.join(train_dir, 'covid')
train_normal_dir = os.path.join(train_dir, 'normal')
train_pneumonia_dir = os.path.join(train_dir, 'pneumonia')

val_covid_dir = os.path.join(val_dir, 'covid')
val_normal_dir = os.path.join(val_dir, 'normal')
val_pneumonia_dir = os.path.join(val_dir, 'pneumonia')

test_covid_dir = os.path.join(test_dir, 'covid')
test_normal_dir = os.path.join(test_dir, 'normal')
test_pneumonia_dir = os.path.join(test_dir, 'pneumonia')

INPUT_SIZE = 100
BATCH_SIZE = 54


""" Investigate train - val - test datasets """

train_batches = ImageDataGenerator(rescale = 1 / 255.).flow_from_directory(train_dir,
                                                         target_size=(INPUT_SIZE,INPUT_SIZE),
                                                         class_mode='categorical',
                                                         shuffle=True,
                                                         seed=42,
                                                         batch_size=BATCH_SIZE)

val_batches = ImageDataGenerator(rescale = 1 / 255.).flow_from_directory(val_dir,
                                                         target_size=(INPUT_SIZE,INPUT_SIZE),
                                                         class_mode='categorical',
                                                         shuffle=True,
                                                         seed=42,
                                                         batch_size=BATCH_SIZE)

test_batches = ImageDataGenerator(rescale = 1 / 255.).flow_from_directory(test_dir,
                                                         target_size=(INPUT_SIZE,INPUT_SIZE),
                                                         class_mode='categorical',
                                                         shuffle=False,
                                                         seed=42,
                                                         batch_size=BATCH_SIZE)

num_covid_train = int(len(os.listdir(train_covid_dir)))
num_normal_train = int(len(os.listdir(train_normal_dir)))
num_pneumonia_train = int(len(os.listdir(train_pneumonia_dir)))

num_covid_val = int(len(os.listdir(val_covid_dir)))
num_normal_val = int(len(os.listdir(val_normal_dir)))
num_pneumonia_val = int(len(os.listdir(val_pneumonia_dir)))

num_covid_test = int(len(os.listdir(test_covid_dir)))
num_normal_test = int(len(os.listdir(test_normal_dir)))
num_pneumonia_test = int(len(os.listdir(test_pneumonia_dir)))

print('The dataset contains:')
print('\u2022 {} training images'.format(num_covid_train + num_normal_train + num_pneumonia_train))
print('\u2022 {} validation images'.format(num_covid_val + num_normal_val + num_pneumonia_val))
print('\u2022 {} test images'.format(num_covid_test + num_normal_test + num_pneumonia_test))

print('\nThe training set contains:')
print('\u2022 {} covid images'.format(num_covid_train))
print('\u2022 {} normal images'.format(num_normal_train))
print('\u2022 {} pneumonia images'.format(num_pneumonia_train))

print('\nThe validation set contains:')
print('\u2022 {} covid images'.format(num_covid_val))
print('\u2022 {} normal images'.format(num_normal_val))
print('\u2022 {} pneumonia images'.format(num_pneumonia_val))

print('\nThe test set contains:')
print('\u2022 {} covid images'.format(num_covid_test))
print('\u2022 {} normal images'.format(num_normal_test))
print('\u2022 {} pneumonia images'.format(num_pneumonia_test))

MODEL_FNAME = 'embedding_network.h5'

tf.compat.v1.reset_default_graph()

if not os.path.exists(MODEL_FNAME):

    """ Create the model """
    
    base_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(INPUT_SIZE,INPUT_SIZE,3))
    
    print("Number of layers in the base model: ", len(base_model.layers))
    
    base_model.trainable = True
    
    base_model.summary()
    
    last_output = base_model.output
    
    x = Flatten()(last_output)
    x = Dense(3, activation='softmax')(x)
    
    embedding_network = Model(inputs=[base_model.input], outputs=[x])
    
    optimizer = Adam(learning_rate=0.00001) 
    embedding_network.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    embedding_network.summary()
    
    """ callbacks """
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    
    checkpointer = ModelCheckpoint(filepath='embedding_network.h5', verbose=1, 
                                   save_best_only=True)
            
    """ Train the model """
    
    history = embedding_network.fit(
        train_batches,
        validation_data = val_batches,
        epochs = 140,
        verbose = 1,
        shuffle = True,
        callbacks = [early_stopping, checkpointer]
    )
    
    """ plot the train and val accuracies """
    # Plot the accuracy
    utils.plt_metric(history=history.history, metric="acc", title="Model accuracy")
    
    # Plot the constrastive loss
    utils.plt_metric(history=history.history, metric="loss", title="Constrastive Loss")

    
    print("End of Training, the model is saved to", MODEL_FNAME)  
    tf.keras.backend.clear_session()
else:
    """ Test the model """
    model = tf.keras.models.load_model(MODEL_FNAME)
    model.summary()
    
    y_test = test_batches.classes
    
    #Confution Matrix and Classification Report
    Y_pred = model.predict_generator(test_batches, (num_covid_test + num_normal_test + num_pneumonia_test) // BATCH_SIZE+1)
    y_pred = np.argmax(Y_pred, axis=1)
    
    class_labels = list(test_batches.class_indices.keys())   
    
    cm = confusion_matrix(test_batches.classes, y_pred)    
    cm_display = ConfusionMatrixDisplay(cm, class_labels).plot()
    
    # results = model.evaluate_generator(test_batches)
    print("\nEvaluate on test data")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted'))
    print("Recall:", recall_score(y_test, y_pred, average='weighted'))
    
tf.keras.backend.clear_session()