#!/usr/bin/env python

import keras
import pred
import numpy as np
import matplotlib.pyplot as plt


def main():
    START = 4000
    NIMAGES = 10

    # XXX: Load the test dataset
    pred.TSTEPS = 1
    valX, valY = pred.load_image_for_keras(START=START, NUM_IMAGES=NIMAGES,
                                           TSTEP=1)
    print(valX.shape, valY.shape)
    valX = valX.reshape(1, *valX.shape)
    print(valX.shape, valY.shape)

    # XXX: Load the model
    m1 = keras.saving.load_model('model2.keras')
    print(m1.summary())

    # XXX: Make a prediction for 10 images
    out = m1(valX, training=False).numpy().reshape(*valY.shape)
    valX = valX.reshape(*valY.shape)
    for x, y, yp in zip(valX, valY, out):
        fig, (xa, ax1, ax2) = plt.subplots(1, 3)
        y *= 255
        yp *= 255
        x *= 255
        y = np.uint8(y)
        yp = np.uint8(yp)
        x = np.uint8(x)
        xa.title.set_text('Input')
        xa.imshow(x)
        ax1.title.set_text('Ground truth')
        ax1.imshow(y)
        ax2.title.set_text('Predicted')
        ax2.imshow(yp)
        plt.show()
        plt.close(fig)


if __name__ == '__main__':
    main()
