"""
    runner.py
    Nicholas S. Bradford
    21 April 2017
    CS 529 - Machine Learning
    Final Project: End to end autonomous steering for self-driving cars.
    Focus Area: Convolutional Neural Networks.
    Advanced Topic: Batch Normalization

    Objective: Given an RGB image, predict steering wheel angle.
    Cost: mean squared error



    Batch normalization:
        https://keras.io/layers/normalization/
"""


from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import losses

# =================================================================================================

def create_model_dropout(input_shape):
    """ Architecture: 
    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(40, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))
    # model.add(Dropout(0.5))
    return model

def create_model_batch_normalization(input_shape):
    """ Normalize the activations of the previous layer at each batch, 
        i.e. applies a transformation that maintains the mean activation 
        close to 0 and the activation standard deviation close to 1.
    """
    # BatchNormalization(axis=-1,
    #                     momentum=0.99,
    #                     epsilon=0.001,
    #                     center=True,
    #                     scale=True,
    #                     beta_initializer='zeros',
    #                     gamma_initializer='ones',
    #                     moving_mean_initializer='zeros',
    #                     moving_variance_initializer='ones',
    #                     beta_regularizer=None,
    #                     gamma_regularizer=None,
    #                     beta_constraint=None,
    #                     gamma_constraint=None)
    pass


def create_model_basic(input_shape):
    """ Architecture: 
    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(40, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))
    # model.add(Dropout(0.5))
    return model

def steering_experiment(batch_size=128, epochs=1):
    img_rows, img_cols = 28, 28 # input image dimensions
    input_shape, x_train, x_test, y_train, y_test = get_train_test_data(img_rows, img_cols, num_classes)
    model = create_model_basic(input_shape)

    # Adam optimizer:
    # original paper https://arxiv.org/abs/1412.6980
    # keras https://keras.io/optimizers/#adam
    # adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


# =================================================================================================
# Keras basic MNIST-CNN example:
#     https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

def get_train_test_data(img_rows, img_cols, num_classes):
    """ TODO separate training data into training and test sets 
        Might be able to use scikit-learn function?
    """
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # reshape according to the backend settings; 
    # image_data_format() is 'channels_first' or 'channels_last'
    if keras.backend.image_data_format() == 'channels_first': 
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return input_shape, x_train, x_test, y_train, y_test


def run_experiment(batch_size=128, num_classes=10, epochs=1):
    img_rows, img_cols = 28, 28 # input image dimensions
    input_shape, x_train, x_test, y_train, y_test = get_train_test_data(img_rows, img_cols, num_classes)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss: {} \t Test Accuracy {} '.format(score[0], score[1]))


if __name__ == '__main__':
    run_experiment()
