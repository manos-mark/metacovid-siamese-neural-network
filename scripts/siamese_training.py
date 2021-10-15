# -*- coding: utf-8 -*-
import scripts.utils as utils

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

print('Using:')
print('\t\u2022 TensorFlow version:', tf.__version__)
print('\t\u2022 tf.keras version:', tf.keras.__version__)
print('\t\u2022 Running on GPU' if tf.test.is_gpu_available() else '\t\u2022 GPU device not found. Running on CPU')


MODEL_FNAME = 'embedding_network.h5'
        
basedir = os.path.join("dataset", "siamese") 

train_image_list, train_y_list = utils.load_images(basedir, 'train')
print("The train set contains",len(train_image_list)) 

valid_image_list, valid_y_list = utils.load_images(basedir, 'validation')   
print("The valid set contains", len(valid_image_list))  

# test_image_list, test_y_list = load_images(basedir, 'test')   
# print("The test set contains", len(test_image_list))  


# make train pairs
pairs_train, labels_train = utils.make_pairs(train_image_list, train_y_list)

# make validation pairs
pairs_val, labels_val = utils.make_pairs(valid_image_list, valid_y_list)

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


input_1 = Input((100,100,3))
input_2 = Input((100,100,3))


embedding_network = tf.keras.models.load_model(MODEL_FNAME)
embedding_network.trainable = False

# add here as the output of embedding network Towards the end of the pretrained 
# model we add a flatten layer which is followed by a dense layer with 5120 
# neurons, sigmoid activation function, and L2 kernel regularizer

# tower_1 = embedding_network(input_1)
# tower_2 = embedding_network(input_2)

# merge_layer = utils.manhattan_distance([tower_1, tower_2])
# output_layer = Dense(1, activation="sigmoid")(merge_layer)

# siamese = Model(inputs=[input_1, input_2], outputs=[output_layer])
# siamese.summary()

# """ callbacks """

# # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.000001)
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
# checkpointer = ModelCheckpoint(filepath='siamese_network.h5', verbose=1, 
#                                 save_best_only=True)


# x_train_1_tensor = tf.convert_to_tensor(x_train_1, dtype=tf.float32)
# x_train_2_tensor = tf.convert_to_tensor(x_train_2, dtype=tf.float32)

# x_val_1_tensor = tf.convert_to_tensor(x_val_1, dtype=tf.float32)
# x_val_2_tensor = tf.convert_to_tensor(x_val_2, dtype=tf.float32)

# """ train the model """

# optimizer = Adam(learning_rate=0.0001)
# siamese.compile(loss=utils.loss(1), optimizer=optimizer, metrics=["accuracy"])
                 
# siamese.summary()
# history = siamese.fit([x_train_1_tensor, x_train_2_tensor],
#     labels_train,
#     validation_data=([x_val_1_tensor, x_val_2_tensor], labels_val),
#     batch_size=10,
#     epochs=175,   # 175 for contrastive 100 for cross ent
#     callbacks = [early_stopping]
# )

