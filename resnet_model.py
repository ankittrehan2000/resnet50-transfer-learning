datasetdir = '/Users/ankittrehan/Desktop/'
import os
os.chdir(datasetdir)

import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from keras.applications.resnet50 import preprocess_input, ResNet50, decode_predictions
from keras.layers import Dropout, Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras import Sequential
import keras.backend as K
import keras

batch_size = 32

def generators(shape, preprocessing):
    #create training and validation tests
    image_data_gen = ImageDataGenerator(
        preprocessing_function = preprocessing,
        horizontal_flip = True,
        validation_split = 0.1
    )

    height, width = shape

    train_dataset = image_data_gen.flow_from_directory(
        os.getcwd(),
        target_size = (height, width),
        classes = ('egg', 'noegg'),
        #shuffle=True,
        batch_size = batch_size, 
        subset = 'training'
    )

    val_dataset = image_data_gen.flow_from_directory(
        os.getcwd(),
        target_size = (height, width),
        #shuffle=True,
        classes = ('egg', 'noegg'),
        batch_size = batch_size,
        subset = 'validation'
    )

    return train_dataset, val_dataset

def plot_history(history, yrange):
    #plots for accuracy
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc)
    plt.plot(epochs, val_acc)
    plt.title("Training and validation accuracy")
    plt.ylim(yrange)
    plt.figure()
    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.title("Training and validation loss")
    plt.show()

dataset, val_dataset = generators((224, 224), preprocessing=keras.applications.resnet50.preprocess_input)
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#model.summary()
#batch normalization
for layer in model.layers:
    '''if hasattr(layer, 'moving_mean') and hasattr(layer, 'moving_variance'):
        layer.trainable = True
        K.eval(K.update(layer.moving_mean, K.zeros_like(layer.moving_mean)))
        K.eval(K.update(layer.moving_variance, K.zeros_like(layer.moving_variance)))
    else:
        layer.trainable = False'''
    layer.trainable = True
dataset, val_dataset = generators((224, 224), preprocessing=keras.applications.resnet50.preprocess_input)

added_layer_model = keras.layers.Flatten()(model.output)
added_layer_model = keras.layers.Dropout(0.5)(added_layer_model)
added_layer_model = keras.layers.Dense(512, activation='relu')(added_layer_model)
added_layer_model = keras.layers.Dropout(0.3)(added_layer_model)
added_layer_model = keras.layers.Dense(256, activation='relu')(added_layer_model)
#this was due to overtraining the model
added_layer_model = keras.layers.Dense(64, activation='relu')(added_layer_model)
added_layer_model = keras.layers.Dense(32, activation='relu')(added_layer_model)

prediction_layer_model = keras.layers.Dense(2, activation='softmax')(added_layer_model)

full_resnet_model = keras.models.Model(input = model.input, outputs = prediction_layer_model)

full_resnet_model.compile(
    loss='binary_crossentropy',
    optimizer = keras.optimizers.SGD(lr = 0.01),
    metrics=['acc']
)

history = full_resnet_model.fit_generator(
    dataset,
    validation_data= val_dataset,
    epochs = 5,
    steps_per_epoch = 24,
)

keras.models.save_model(full_resnet_model, 'SLFModel.pb')

plot_history(history, yrange=(0.9, 1))
