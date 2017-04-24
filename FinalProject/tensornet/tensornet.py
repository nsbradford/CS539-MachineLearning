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
        https://keras.io/layers/normalization/
"""

import os
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization

from dataload import datagen

training_files = ['data/camera/2016-01-30--11-24-51.h5']
validation_files = ['data/camera/2016-01-31--19-19-25.h5']

def gen_training():
    for tup in datagen(training_files): # img, steering angle, speed
        X, Y, _ = tup # drop the speed
        Y = Y[:, -1]
        if X.shape[1] == 1:  # no temporal context
            X = X[:, -1]
        yield X, Y


def gen_validation():
    for tup in datagen(validation_files): # img, steering angle, speed
        X, Y, _ = tup # drop the speed
        Y = Y[:, -1]
        if X.shape[1] == 1:  # no temporal context
            X = X[:, -1]
        yield X, Y

# def gen_training():
#     return gen(is_training=True)


# def gen_validation():
#     return gen(is_training=False)

# def gen_training():
#     for tup in gen(is_training=True):
#         yield tup

# def gen_validation():
#     for tup in gen(is_training=False):
#         yield tup


def create_model_basic():
    """
        Model from Comma.ai:
            https://github.com/commaai/research/blob/master/train_steering_model.py
        Meta-params:
            Adam optimizer (default params) and MSE loss function.
        Architecture:
            Normalization layer -> data in [-1, 1] range (done in NN to allow GPU use)
            Conv2D: 16@8x8, stride (4, 4), ELU activation
            Conv2D: 32@5x5, stride (2, 2), ELU activation
            Conv2D: 64@5x5, stride (2, 2), ELU activation
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
            Feedforward: 512, ELU activation dropout rate=0.2
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


def main(N_EPOCHS=1, BATCHES_PER_EPOCH=2000):
    """ batch size of 256 * 10000 batches, ~50,000 training frames
    """
    # model = create_model_dropout()
    model = create_model_basic()
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
    