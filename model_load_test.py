#!/usr/bin/env python

import keras
import pred
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import glob


def date_to_num(date, dd='./figs'):
    ff = sorted(glob.glob(dd+'/*.png'))
    count = 0
    for i in ff:
        if i.split('/')[-1].split('_')[0] == date:
            break
        count += 1
    return count


def num_to_date(num, dd='./figs'):
    ff = sorted(glob.glob(dd+'/*.png'))
    return ff[num].split('/')[-1].split('_')[0]


def load_image(num, dd='./figs', RESIZE_FACTOR=8):
    ff = sorted(glob.glob(dd+'/*.png'))
    img = load_img(ff[num])
    w, h = img.size
    img = img.resize((w//RESIZE_FACTOR, h//RESIZE_FACTOR))
    img_gray = img.convert('L')
    img_array = img_to_array(img_gray)
    return img_array


def main():
    START = date_to_num('20230221')
    print("START:", START)
    NIMAGES = 20

    # XXX: Load the test dataset
    pred.TSTEPS = 5
    valX, valY, Ydates = pred.load_image_for_keras(START=START,
                                                   NUM_IMAGES=NIMAGES,
                                                   TSTEP=pred.TSTEPS)
    print(valX.shape, valY.shape)
    valX = valX.reshape(valX.shape[0]//pred.TSTEPS, pred.TSTEPS,
                        *valX.shape[1:])
    print(valX.shape, valY.shape)

    # XXX: Load the model
    m1 = keras.saving.load_model(
        './modelcr_bs_2_ts_%s_filters_64.keras' % (pred.TSTEPS))
    print(m1.summary())

    # XXX: Make a prediction for 10 images
    out = m1(valX, training=False).numpy()
    print(out.shape)
    for y, yd, yp in zip(valY, Ydates, out):
        y *= 255
        yp *= 255
        # x *= 255
        y = np.uint8(y)
        yp = np.uint8(yp)
        # x = np.uint8(x)
        # xa.title.set_text('Input')
        # xa.imshow(x)
        ynum = date_to_num(yd)
        pyd = num_to_date(ynum-1)
        print(int(yd), int(pyd))
        if (int(yd) - int(pyd)) == 1:
            # XXX: Then we can get this figure
            ximg = load_image(ynum-1)
            fig, axs = plt.subplots(2, 2, sharex=True)
            axs[0, 0].title.set_text('Ground truth date: ' + yd)
            axs[0, 0].imshow(y, cmap='hot')
            axs[0, 1].title.set_text('Predicted date: ' + yd)
            axs[0, 1].imshow(yp, cmap='hot')
            dimgt = y - ximg
            axs[1, 0].imshow(dimgt, cmap='PuBu')
            axs[1, 0].title.set_text('True diff : %s_%s' % (yd, pyd))
            dimg = yp - ximg
            axs[1, 1].imshow(dimg, cmap='PuBu')
            axs[1, 1].title.set_text('diff dates :%s-%s' % (yd, pyd))
            plt.show()
            plt.close(fig)


if __name__ == '__main__':
    main()
    # dir_to_num()
