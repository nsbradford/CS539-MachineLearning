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

    Note: no max pooling used!

    ELU activation:
        original paper https://arxiv.org/abs/1511.07289
    Adam optimizer:
        original paper https://arxiv.org/abs/1412.6980
        keras https://keras.io/optimizers/#adam
    Batch normalization:
        keras docs: https://keras.io/layers/normalization/
        original paper: https://arxiv.org/pdf/1502.03167.pdf
"""

import os
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization

from dataload import datagen



def gen_training():
    training_files = ['data/camera/2016-01-30--11-24-51.h5']
    for tup in datagen(training_files): # img, steering angle, speed
        X, Y, _ = tup # drop the speed
        Y = Y[:, -1]
        if X.shape[1] == 1:  # no temporal context
            X = X[:, -1]
        yield X, Y


def gen_validation():
    validation_files = ['data/camera/2016-01-31--19-19-25.h5']
    for tup in datagen(validation_files): # img, steering angle, speed
        X, Y, _ = tup # drop the speed
        Y = Y[:, -1]
        if X.shape[1] == 1:  # no temporal context
            X = X[:, -1]
        yield X, Y


def create_model_basic():
    """
        Model from Comma.ai:
            https://github.com/commaai/research/blob/master/train_steering_model.py
        Meta-params:
            Adam optimizer (default params) and MSE loss function.
        Architecture:
            Normalization layer -> data in [-1, 1] range (done in NN to allow GPU use)
            Conv2D: 16, 8x8, stride (4, 4), ELU activation
            Conv2D: 32, 5x5, stride (2, 2), ELU activation
            Conv2D: 64, 5x5, stride (2, 2), ELU activation
            Feedforward: 512, ELU activation
            Feedforward: 1 (output layer)
    """
    ch, row, col = 3, 160, 320  # camera format
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
                        input_shape=(ch, row, col),
                        output_shape=(ch, row, col)))
    model.add(Conv2D(16, (8, 8), strides=(4, 4), padding="same"))
    model.add(ELU())
    model.add(Conv2D(32, (5, 5), strides=(2, 2), padding="same"))
    model.add(ELU())
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same"))
    model.add(Flatten())
    model.add(ELU())
    model.add(Dense(512))
    model.add(ELU())
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model


def create_model_nvidia():
    """
        Model from NVIDIA paper:
            https://arxiv.org/abs/1604.07316
        Meta-params:
            Adam optimizer (default params) and MSE loss function.
        Architecture:
            Normalization layer -> data in [-1, 1] range (done in NN to allow GPU use)
            Conv2D: 24, 5x5, stride (2, 2), "valid" instead of padded, ELU activation
            Conv2D: 36, 5x5, stride (2, 2), "valid" instead of padded, ELU activation
            Conv2D: 48, 5x5, stride (2, 2), "valid" instead of padded, ELU activation
            Conv2D: 64, 3x3, "valid" instead of padded, ELU activation
            Conv2D: 64, 3x3, "valid" instead of padded, ELU activation
            Feedforward: 100, ELU activation
            Feedforward: 50, ELU activation
            Feedforward: 10, ELU activation
            Feedforward: 1 (output layer)

        Note: we have to use dim_ordering='th' for the convolutional layers to accept
            the input in this format, despite using TensorFlow and not Theano.
    """
    ch, row, col = 3, 160, 320  # camera format
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
                        input_shape=(ch, row, col),
                        output_shape=(ch, row, col)))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), padding="valid", activation='elu', dim_ordering='th'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), padding="valid", activation='elu', dim_ordering='th'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), padding="valid", activation='elu', dim_ordering='th'))
    model.add(Conv2D(64, (3, 3), padding="valid", activation='elu', dim_ordering='th'))
    model.add(Conv2D(64, (3, 3), padding="valid", activation='elu', dim_ordering='th'))
    model.add(Flatten()) 
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model


def create_model_dropout():
    """
        Model from Comma.ai:
            https://github.com/commaai/research/blob/master/train_steering_model.py
        Meta-params:
            Adam optimizer (default params) and MSE loss function.
        Architecture:
            Normalization layer to get all data in [-1, 1] range (done in NN to allow GPU use)
            Conv2D: 16@8x8, stride (4, 4), ELU activation
            Conv2D: 32@5x5, stride (2, 2), ELU activation
            Conv2D: 64@5x5, stride (2, 2), ELU activation, dropout rate=0.2
            Feedforward: 512, ELU activation, dropout rate=0.2
            Feedforward: 1 (output layer)
    """
    ch, row, col = 3, 160, 320  # camera format
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
                        input_shape=(ch, row, col),
                        output_shape=(ch, row, col)))
    model.add(Conv2D(16, (8, 8), strides=(4, 4), padding="same"))
    model.add(ELU())
    model.add(Conv2D(32, (5, 5), strides=(2, 2), padding="same"))
    model.add(ELU())
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model


def create_model_batch_normalization_simple():
    """ Normalize the activations of the previous layer at each batch, 
        i.e. applies a transformation that maintains the mean activation 
        close to 0 and the activation standard deviation close to 1.

        Architecture:
            Same as Comma.AI (simple), except add batch normalization in the
                feedforward layer
    """
    ch, row, col = 3, 160, 320  # camera format
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
                        input_shape=(ch, row, col),
                        output_shape=(ch, row, col)))
    model.add(Conv2D(16, (8, 8), strides=(4, 4), padding="same", activation='elu'))
    model.add(Conv2D(32, (5, 5), strides=(2, 2), padding="same", activation='elu'))
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same", activation='elu'))
    model.add(Flatten())
    model.add(Dense(512, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model


def create_model_batch_normalization_nvidia():
    """ Normalize the activations of the previous layer at each batch, 
        i.e. applies a transformation that maintains the mean activation 
        close to 0 and the activation standard deviation close to 1.

        Architecture:
            Same as NVidia model, except add batch normalization in the
                feedforward layers.

        Note: we have to use dim_ordering='th' for the convolutional layers to accept
            the input in this format, despite using TensorFlow and not Theano.
    """
    ch, row, col = 3, 160, 320  # camera format
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
                        input_shape=(ch, row, col),
                        output_shape=(ch, row, col)))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), padding="valid", activation='elu', dim_ordering='th'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), padding="valid", activation='elu', dim_ordering='th'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), padding="valid", activation='elu', dim_ordering='th'))
    model.add(Conv2D(64, (3, 3), padding="valid", activation='elu', dim_ordering='th'))
    model.add(Conv2D(64, (3, 3), padding="valid", activation='elu', dim_ordering='th'))
    model.add(Flatten()) 
    model.add(Dense(100, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dense(50, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model


def main(N_EPOCHS=1, BATCHES_PER_EPOCH=4000):
    """ batch size of 256 * 10000 batches, ~50,000 training frames
    """
    # model = create_model_basic()
    # model = create_model_nvidia()
    model = create_model_batch_normalization_simple() 
    # model = create_model_batch_normalization_nvidia()
    # model = create_model_dropout()
    
    model.fit_generator(
        gen_training(),
        samples_per_epoch=BATCHES_PER_EPOCH,
        nb_epoch=N_EPOCHS,
        validation_data=gen_validation(),
        nb_val_samples=1000
    )
    print("Saving model weights and configuration file.")
    OUTPUT_DIR = './output_models/steering_model/'
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    model.save_weights(OUTPUT_DIR + 'steering_angle.keras', True)
    with open(OUTPUT_DIR + 'steering_angle.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)

    # score = model.evaluate(x_test, y_test, verbose=0)
    # print('Test loss: {} \t Test Accuracy {} '.format(score[0], score[1]))


if __name__ == "__main__":
    main()
    