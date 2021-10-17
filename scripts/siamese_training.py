# -*- coding: utf-8 -*-
import scripts.utils as utils

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

import numpy as np
import os
from tensorflow.keras.regularizers import l2

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

print('Using:')
print('\t\u2022 TensorFlow version:', tf.__version__)
print('\t\u2022 tf.keras version:', tf.keras.__version__)
print('\t\u2022 Running on GPU' if tf.test.is_gpu_available() else '\t\u2022 GPU device not found. Running on CPU')


basedir = os.path.join("dataset", "siamese") 

train_image_list, train_y_list = utils.load_images(basedir, 'train', (100,100))
print("The train set contains",len(train_image_list)) 

valid_image_list, valid_y_list = utils.load_images(basedir, 'validation', (100,100))   
print("The valid set contains", len(valid_image_list))  

# make train pairs
pairs_train, labels_train = utils.make_pairs(train_image_list, train_y_list)

# make validation pairs
pairs_val, labels_val = utils.make_pairs(valid_image_list, valid_y_list)

x_train_1 = pairs_train[:, 0]  
x_train_2 = pairs_train[:, 1]
print("number of pairs for training", np.shape(x_train_1)[0]) 

x_val_1 = pairs_val[:, 0] 
x_val_2 = pairs_val[:, 1]
print("number of pairs for validation", np.shape(x_val_1)[0]) 

# utils.visualize(pairs_train[:-1], labels_train[:-1], to_show=4, num_col=4)

tf.compat.v1.reset_default_graph()

SIAMESE_MODEL_FNAME = 'siamese_network.h5'
EMBEDDING_MODEL_FNAME = 'embedding_network.h5'

if not os.path.exists(SIAMESE_MODEL_FNAME):
    
    """ L1 mistance - manhattan """
    def manhattan_distance(vects):
        x, y = vects
        return K.sum(K.abs(x-y), axis=1, keepdims=True)

    input_1 = Input((100,100,3))
    input_2 = Input((100,100,3))
    
    embedding_network = tf.keras.models.load_model(EMBEDDING_MODEL_FNAME)
    embedding_network.trainable = False
    
    model = tf.keras.Sequential() 
    for layer in embedding_network.layers: # go through until last layer 
        model.add(layer) 
    
    model.add(Flatten(name='flat'))
    model.add(Dense(5120, name='den', activation='sigmoid', kernel_regularizer='l2')) 
     
    output_1 = model(input_1) 
    output_2 = model(input_2) 
     
    merge_layer = Lambda(manhattan_distance)([output_1, output_2]) 
    output_layer = Dense(1, activation="sigmoid")(merge_layer) 
    siamese = Model(inputs=[input_1, input_2], outputs=output_layer) 
    siamese.summary()
    
    """ callbacks """
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.000001)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    
    checkpointer = ModelCheckpoint(filepath='siamese_network.h5', verbose=1, 
                                    save_best_only=True)
    
    """ train the model """
    
    optimizer = Adam(learning_rate=0.0001)
    siamese.compile(loss=utils.loss(1), optimizer=optimizer, metrics=["accuracy"])
    # siamese.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    
    siamese.summary()
    history = siamese.fit([x_train_1, x_train_2],
        labels_train,
        validation_data=([x_val_1, x_val_2], labels_val),
        batch_size=1,
        # steps_per_epoch=10,
        epochs=175,   # 175 for contrastive 100 for cross ent
        callbacks = [checkpointer, early_stopping, reduce_lr]
    )
    
    # Plot the accuracy
    utils.plt_metric(history=history.history, metric="acc", title="Model accuracy")
    
    # Plot the constrastive loss
    utils.plt_metric(history=history.history, metric="loss", title="Constrastive Loss")
    
else:
    """ Test the model """
    
    test_image_list, test_y_list = utils.load_images(basedir, 'test', (100,100))   
    print("The test set contains", len(test_image_list))  
    
    # make test pairs
    pairs_test, labels_test = utils.make_pairs(test_image_list, test_y_list)
    
    x_test_1 = pairs_test[:, 0] 
    x_test_2 = pairs_test[:, 1]
    print("number of pairs for test", np.shape(x_test_1)[0]) 

    model = tf.keras.models.load_model(SIAMESE_MODEL_FNAME)
    model.summary()
    
    results = siamese.evaluate([x_test_1, x_test_2], labels_test)
    print("test loss, test acc:", results)

    
    # y_test = test_batches.classes
    
    # #Confution Matrix and Classification Report
    # Y_pred = model.predict_generator(test_batches, (num_covid_test + num_normal_test + num_pneumonia_test) // BATCH_SIZE+1)
    # y_pred = np.argmax(Y_pred, axis=1)
    
    # class_labels = list(test_batches.class_indices.keys())   
    
    # cm = confusion_matrix(test_batches.classes, y_pred)    
    # cm_display = ConfusionMatrixDisplay(cm, class_labels).plot()
    
    # # results = model.evaluate_generator(test_batches)
    # print("\nEvaluate on test data")
    # print("Accuracy:", accuracy_score(y_test, y_pred))
    # print("Precision:", precision_score(y_test, y_pred, average='weighted'))
    # print("Recall:", recall_score(y_test, y_pred, average='weighted'))

tf.keras.backend.clear_session()