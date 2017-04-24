"""
    Demo from Comma.ai:
        https://github.com/commaai/research/blob/master/train_steering_model.py
"""

import os
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Conv2D

from dataload import datagen

# def gen(hwm, host, port):
#     for tup in client_generator(hwm=hwm, host=host, port=port):
#         X, Y, _ = tup
#         Y = Y[:, -1]
#         if X.shape[1] == 1:  # no temporal context
#             X = X[:, -1]
#         yield X, Y

def gen():
    for tup in datagen(): # img, steering angle, speed
        X, Y, _ = tup # drop the speed
        Y = Y[:, -1]
        if X.shape[1] == 1:  # no temporal context
            X = X[:, -1]
        yield X, Y


def get_model(time_len=1):
    """
        Architecture:
            Conv2D: 16@8x8
            Conv2D: 16@8x8
            Conv2D: 16@8x8
            Feedforward: 512 (with dropout)
            Feedforward: 1 (with dropout)
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


def main(N_EPOCHS=1, BATCHES_PER_EPOCH=10000):
    model = get_model()
    model.fit_generator(
        gen(),
        samples_per_epoch=BATCHES_PER_EPOCH,
        nb_epoch=N_EPOCHS,
        validation_data=gen(), # TODO add validation set
        nb_val_samples=1000
    )
    print("Saving model weights and configuration file.")
    OUTPUT_DIR = './output_models/steering_model/'
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    model.save_weights(OUTPUT_DIR + 'steering_angle.keras', True)
    with open(OUTPUT_DIR + 'steering_angle.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)


if __name__ == "__main__":
    main()
    