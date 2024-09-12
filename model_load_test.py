#!/usr/bin/env python

import tensorflow as tf
from tensorflow import keras 
import pred
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import glob


def date_to_num(date, dd='.\\figs'):
    ff = sorted(glob.glob(dd+'\\*.npy'))
    count = 0
    for i in ff:
        if i.split('\\')[-1].split('.')[0] == date:
            break
        count += 1
    return count


def num_to_date(num, dd='.\\figs'):
    ff = sorted(glob.glob(dd+'\\*.npy'))
    return ff[num].split('\\')[-1].split('.')[0]


def load_image(num, dd='.\\figs'):
    ff = sorted(glob.glob(dd+'\\*.npy'))
    img = np.load(ff[num])
    return img


def main():
    # XXX: Important date: 20201014

    START = date_to_num('20201014')
    print("START:", START)
    NIMAGES = 30

    # XXX: Load the test dataset
    pred.TSTEPS = 5
    bs = 5
    nf = 64

    valX, valY, Ydates = pred.load_data_for_keras(START=START,
                                                  NUM_IMAGES=NIMAGES,
                                                  TSTEP=pred.TSTEPS)
    print(valX.shape, valY.shape)
    valX = valX.reshape(valX.shape[0]//pred.TSTEPS, pred.TSTEPS,
                        *valX.shape[1:])
    print(valX.shape, valY.shape)

    # XXX: Load the model
    m1 = keras.models.load_model(
        './modelcr_bs_%s_ts_%s_filters_%s_%s.keras' % (bs, pred.TSTEPS, nf, tf.__version__))
    print(m1.summary())

    # XXX: The moneyness
    MS = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)
    # XXX: The term structure
    TS = np.array([i/pred.DAYS
                   for i in
                   range(pred.LT, pred.UT+pred.TSTEP, pred.TSTEP)])

    # XXX: Make a prediction for 10 images
    out = m1(valX, training=False).numpy()

    for y, yd, yp in zip(valY, Ydates, out):
        y *= 100
        yp *= 100
        ynum = date_to_num(yd)
        pyd = num_to_date(ynum-1)
        if (int(yd) - int(pyd)) == 1:
            # XXX: Then we can get this figure
            fig, axs = plt.subplots(1, 3,
                                    subplot_kw=dict(projection='3d'))
            (ax.view_init(30, 45) for ax in axs)
            axs[0].title.set_text('Truth: ' + yd)
            # XXX: Make the y dataframe
            ydf = list()
            for cm, m in enumerate(MS):
                for ct, t in enumerate(TS):
                    ydf.append([m, t, y[cm, ct]])
            ydf = np.array(ydf)
            axs[0].plot_trisurf(ydf[:, 0], ydf[:, 1], ydf[:, 2],
                                cmap='jet', linewidth=0.1,
                                antialiased=True)
            min_z = ydf[:, 2].min()
            axs[0].tricontourf(ydf[:, 0], ydf[:, 1], ydf[:, 2], zdir='z', offset=min_z, cmap='coolwarm')
            axs[0].set_xlabel('Moneyness')
            axs[0].set_ylabel('Term structure')
            axs[0].set_zlabel('Vol %')
            # axs[0, 0].imshow(np.transpose(y), cmap='hot')
            axs[1].title.set_text('Predicted: ' + yd)
            ypdf = list()
            for cm, m in enumerate(MS):
                for ct, t in enumerate(TS):
                    ypdf.append([m, t, yp[cm, ct]])
            ypdf = np.array(ypdf)
            axs[1].plot_trisurf(ypdf[:, 0], ypdf[:, 1], ypdf[:, 2],
                                cmap='jet', linewidth=0.2,
                                antialiased=True)
            min_z_pred = ypdf[:, 2].min()
            axs[1].tricontourf(ypdf[:, 0], ypdf[:, 1], ypdf[:, 2], zdir='z', offset=min_z_pred, cmap='coolwarm')
            axs[1].set_xlabel('Moneyness')
            axs[1].set_ylabel('Term structure')
            axs[1].set_zlabel('Vol %')
            # axs[0, 1].imshow(np.transpose(yp), cmap='hot')

            # XXX: Previous day' volatility
            ximg = load_image(ynum-1)
            ximg *= 100
            xdf = list()
            for cm, m in enumerate(MS):
                for ct, t in enumerate(TS):
                    xdf.append([m, t, ximg[cm, ct]])
            xdf = np.array(xdf)
            axs[2].plot_trisurf(xdf[:, 0], xdf[:, 1], xdf[:, 2],
                                cmap='jet', linewidth=0.2,
                                antialiased=True)
            min_z_x = xdf[:, 2].min()
            axs[2].tricontourf(xdf[:, 0], xdf[:, 1], xdf[:, 2], zdir='z', offset=min_z_x, cmap='coolwarm')
            axs[2].set_xlabel('Moneyness')
            axs[2].set_ylabel('Term structure')
            axs[2].set_zlabel('Vol %')
            axs[2].title.set_text(pyd)
            # dimgt = y - ximg
            # axs[1, 0].imshow(np.transpose(dimgt), cmap='hot')
            # axs[1, 0].title.set_text('True diff : %s_%s' % (yd, pyd))
            # dimg = yp - ximg
            # axs[1, 1].imshow(np.transpose(dimg), cmap='hot')
            # axs[1, 1].title.set_text('diff dates :%s-%s' % (yd, pyd))
            plt.show()
            plt.close(fig)


if __name__ == '__main__':
    main()
    # dir_to_num()
