# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 15:03:44 2021

@author: aktas
"""
import warnings
warnings.filterwarnings('ignore')

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math

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
BATCH_SIZE = 16


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
print(f'\u2022 %d training images'%(num_covid_train + num_normal_train + num_pneumonia_train))
print(f'\u2022 %d validation images'%(num_covid_val + num_normal_val + num_pneumonia_val))
print(f'\u2022 %d test images'%(num_covid_test + num_normal_test + num_pneumonia_test))

print('\nThe training set contains:')
print(f'\u2022 %d covid images'%(num_covid_train))
print(f'\u2022 %d normal images'%(num_normal_train))
print(f'\u2022 %d pneumonia images'%(num_pneumonia_train))

print('\nThe validation set contains:')
print(f'\u2022 %d covid images'%(num_covid_val))
print(f'\u2022 %d normal images'%(num_normal_val))
print(f'\u2022 %d pneumonia images'%(num_pneumonia_val))

print('\nThe test set contains:')
print(f'\u2022 %d covid images'%(num_covid_test))
print(f'\u2022 %d normal images'%(num_normal_test))
print(f'\u2022 %d pneumonia images'%(num_pneumonia_test))

MODEL_FNAME = 'embedding_network.h5'

if not os.path.exists(MODEL_FNAME):

    """ Create the model """
    
    base_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(INPUT_SIZE,INPUT_SIZE,3))
    
    print("Number of layers in the base model: ", len(base_model.layers))
    
    base_model.trainable = True
    
    base_model.summary()
    
    last_output = base_model.output
    
    x = Dropout(0.5)(last_output)
    
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Flatten()(x)
    x = Dense(3, activation='softmax')(x)
    
    embedding_network = Model(inputs=[base_model.input], outputs=[x])
    
    optimizer = Adam(learning_rate=0.000001) 
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
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
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
    
    cm = confusion_matrix(test_batches.classes, y_pred, normalize='all')    
    cm_display = ConfusionMatrixDisplay(cm, class_labels).plot()
    
    # results = model.evaluate_generator(test_batches)
    print("\nEvaluate on test data")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted'))
    print("Recall:", recall_score(y_test, y_pred, average='weighted'))
    