#!/usr/bin/env python

import pandas as pd
import os
import fnmatch
import zipfile as zip
from sklearn.linear_model import LinearRegression
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import glob
from keras.layers import Input, ConvLSTM2D, BatchNormalization, Conv3D
from keras.models import Model
import keras

# XXX: For plotting only
import matplotlib.pyplot as plt

# XXX: Moneyness Bounds inclusive
LM = 0.9
UM = 1.1
MSTEP = 0.001

# XXX: Tau Bounds inclusive
LT = 14
UT = 366
TSTEP = 1                       # days

DAYS = 365


def preprocess_ivs_df(dfs: dict):
    toret = dict()
    for k in dfs.keys():
        df = dfs[k]
        # XXX: First only get those that have volume > 0
        df = df[df['Volume'] > 0].reset_index(drop=True)
        # XXX: Make the log of K/UnderlyingPrice
        df['m'] = (df['Strike']/df['UnderlyingPrice'])
        # XXX: Moneyness is not too far away from ATM
        df = df[(df['m'] >= LM) & (df['m'] <= UM)]
        # XXX: Make the days to expiration
        df['Expiration'] = pd.to_datetime(df['Expiration'])
        df['DataDate'] = pd.to_datetime(df['DataDate'])
        df['tau'] = (df['Expiration'] - df['DataDate']).dt.days
        # XXX: Only those that are greater than at least 2 weeks ahead
        # and also not too ahead
        df = df[(df['tau'] >= LT) & (df['tau'] <= UT)]
        df['tau'] = df['tau']/DAYS
        df['m2'] = df['m']**2
        df['tau2'] = df['tau']**2
        df['mtau'] = df['m']*df['tau']

        # XXX: This is the final dataframe
        dff = df[['IV', 'm', 'tau', 'm2', 'tau2', 'mtau']]
        toret[k] = dff.reset_index(drop=True)
    return toret


def plot_hmap(ivs_hmap, mrows, tcols):
    for k in ivs_hmap.keys():
        fig, ax = plt.subplots()
        plt.axis('off')
        # print(ivs_hmap[k]/ivs_hmap[k].max())
        # ax.set_xlabel(mrows)
        # ax.set_ylabel(tcols)
        plt.imshow(ivs_hmap[k], cmap='afmhot', interpolation='none')
        plt.savefig('/tmp/figs/{k}_hmap.png'.format(k=k), transparent=True)
        plt.close(fig)
        # break


def plot_ivs(ivs_surface, IVS='IVS', view='XY'):
    for k in ivs_surface.keys():
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        X = ivs_surface[k]['m']
        Y = ivs_surface[k]['tau']
        if IVS == 'IVS':
            Z = ivs_surface[k][IVS]*100
        else:
            Z = ivs_surface[k][IVS]
        # viridis = cm.get_cmap('gist_gray', 256)
        _ = ax.plot_trisurf(X, Y, Z, cmap='afmhot',
                            linewidth=0.2, antialiased=True)
        # ax.set_xlabel('m')
        # ax.set_ylabel('tau')
        # ax.set_zlabel(IVS)
        # ax.view_init(azim=-45, elev=30)
        # ax.invert_xaxis()
        # ax.invert_yaxis()
        if view == 'XY':
            ax.view_init(elev=90, azim=-90)
        elif view == 'XZ':
            ax.view_init(elev=0, azim=-90)
        elif view == 'YZ':
            ax.view_init(elev=0, azim=0)
        ax.axis('off')
        # ax.zaxis.set_major_formatter('{x:.02f}')
        # fig.colorbar(surf, shrink=0.5, aspect=5)
        # plt.show()
        # fig.subplots_adjust(bottom=0)
        # fig.subplots_adjust(top=0.00001)
        # fig.subplots_adjust(right=1)
        # fig.subplots_adjust(left=0)
        plt.savefig('/tmp/figs/{k}_{v}.png'.format(k=k, v=view),
                    bbox_inches='tight', transparent=True, pad_inches=0)
        plt.close(fig)
        # XXX: Convert to gray scale 1 channel only
        # img = Image.open('/tmp/figs/{k}.png'.format(k=k)).convert('LA')
        # img.save('/tmp/figs/{k}.png'.format(k=k))


def load_image_for_keras(dd='./figs', START=0, NUM_IMAGES=1000, NORM=255,
                         RESIZE_FACTOR=4):
    Xs = list()               # Training inputs [0..TEST_IMAGES-1]
    Ys = list()               # Training outputs [1..TEST_IMAGES]
    ff = sorted(glob.glob(dd+'/*.png'))
    # XXX: Load the first TEST_IMAGES for training
    for i in range(START, START+NUM_IMAGES):
        img = load_img(ff[i])   # PIL image
        w, h = img.size
        img = img.resize((w//RESIZE_FACTOR, h//RESIZE_FACTOR))
        img_array = img_to_array(img)/NORM
        Xs += [img_array]
        # XXX: Now do the same thing for the output label image
        img = load_img(ff[i+1])   # PIL image
        w, h = img.size
        img = img.resize((w//RESIZE_FACTOR, h//RESIZE_FACTOR))
        img_array = img_to_array(img)/NORM
        Ys += [img_array]

    # XXX: Convert the lists to np.array
    Xs = np.array(Xs)
    Ys = np.array(Ys)
    np.expand_dims(Xs, axis=-1)
    np.expand_dims(Ys, axis=-1)
    return Xs, Ys


def main(mdir, years, months, instrument, dfs: dict):
    ff = []
    for f in os.listdir(mdir):
        for y in years:
            for m in months:
                # XXX: Just get the year and month needed
                tosearch = "*_{y}_{m}*.zip".format(y=y, m=m)
                if fnmatch.fnmatch(f, tosearch):
                    ff += [f]
                    # XXX: Read the csvs
    for f in ff:
        z = zip.ZipFile(dir+f)
        ofs = [i for i in z.namelist() if 'options_' in i]
        # XXX: Now read just the option data files
        for f in ofs:
            key = f.split(".csv")[0].split("_")[2]
            df = pd.read_csv(z.open(f))
            df = df[df['UnderlyingSymbol'] == instrument].reset_index(
                drop=True)
            dfs[key] = df


def build_gird_and_images(df):
    # XXX: Now fit a multi-variate linear regression to the dataset
    # one for each day.
    df = dict(sorted(df.items()))
    fitted_dict = dict()
    grid = dict()
    scores = list()
    for k in df.keys():
        y = df[k]['IV']
        X = df[k][['m', 'tau', 'm2', 'tau2', 'mtau']]
        reg = LinearRegression(n_jobs=-1).fit(X, y)
        fitted_dict[k] = reg
        scores += [reg.score(X, y)]

        # XXX: Now make the grid
        ss = []
        mms = np.arange(LM, UM+MSTEP, MSTEP)
        tts = [i/DAYS for i in range(LT, UT+TSTEP, TSTEP)]
        for mm in mms:
            for tt in tts:
                # XXX: Make the feature vector
                ss += [[mm, tt, mm**2, tt**2, mm*tt]]
                grid[k] = pd.DataFrame(ss, columns=['m', 'tau', 'm2',
                                                    'tau2', 'mtau'])

    print("average fit score: ", sum(scores)/len(scores))
    # XXX: Now make the smooth ivs surface for each day
    ivs_surf_hmap = dict()
    ivs_surface = dict()
    for k in grid.keys():
        # XXX: This ivs goes m1,t1;m1,t2... then
        # m2,t1;m2,t2,m2,t3.... this is why reshape for heat map as
        # m, t, so we get m rows and t cols. Hence, x-axis is t and
        # y-axis is m.
        pivs = fitted_dict[k].predict(grid[k])
        ivs_surface[k] = pd.DataFrame({'IVS': pivs,
                                       'm': grid[k]['m'],
                                       'tau': grid[k]['tau']})
        ivs_surface[k]['IVS'] = ivs_surface[k]['IVS'].clip(0.01, None)
        # print('IVS len:', len(ivs_surface[k]['IVS']))
        mcount = len(mms)
        tcount = len(tts)
        # print('mcount%s, tcount%s: ' % (mcount, tcount))
        ivs_surf_hmap[k] = ivs_surface[k]['IVS'].values.reshape(mcount,
                                                                tcount)
        # print('ivs hmap shape: ', ivs_surf_hmap[k].shape)

    # XXX: Plot the heatmap
    # plot_hmap(ivs_surf_hmap, mms, tts)

    # XXX: Plot the ivs surface
    # plot_ivs(ivs_surface, view='XY')


def excel_to_images():
    dir = '../HistoricalOptionsData/'
    years = [str(i) for i in range(2011, 2024)]
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
              'August', 'September', 'October', 'November', 'December']
    instrument = ["SPX"]
    dfs = dict()
    # XXX: The dictionary of all the dataframes with the requires
    # instrument ivs samples
    for i in instrument:
        # XXX: Load the excel files
        main(dir, years, months, i, dfs)

        # XXX: Now make ivs surface for each instrument
        df = preprocess_ivs_df(dfs)

        # XXX: Build the images
        build_gird_and_images(df)


def build_keras_model(shape, LR=1e-3):
    inp = Input(shape=shape[1:])
    x = ConvLSTM2D(
        filters=32,
        kernel_size=(7, 7),
        padding="same",
        data_format='channels_last',
        activation='relu',
        return_sequences=True)(inp)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(
        filters=64,
        kernel_size=(5, 5),
        data_format='channels_last',
        padding='same',
        activation='relu',
        return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        data_format='channels_last',
        padding='same',
        activation='relu',
        return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(
        filters=64,
        kernel_size=(1, 1),
        data_format='channels_last',
        padding='same',
        activation='relu',
        return_sequences=True)(x)
    # XXX: The final output layer
    x = Conv3D(
        filters=3, kernel_size=(3, 3, 3), activation="sigmoid",
        padding="same")(x)

    # XXX: The complete model and compiled
    model = Model(inp, x)
    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adam(learning_rate=LR),)
    return model


def keras_model_fit(model, trainX, trainY, valX, valY):
    # Define some callbacks to improve training.
    # early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss",
    #                                                patience=10,
    #              -                                  restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                  patience=5)

    # Define modifiable training hyperparameters.
    epochs = 20
    batch_size = 1

    # Fit the model to the training data.
    history = model.fit(
        trainX,                 # this is not a 5D tensor right now!
        trainY,                 # this is not a 5D tensor right now!
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(valX, valY),
        verbose=1,
        # callbacks=[early_stopping, reduce_lr])
        callbacks=[reduce_lr])
    return history


if __name__ == '__main__':
    # XXX: Excel data to images
    # excel_to_images()

    NIMAGES1 = 3000
    TSTEPS = 10
    START = 0

    # Load, process and learn a ConvLSTM2D network
    trainX, trainY = load_image_for_keras(START=START, NUM_IMAGES=NIMAGES1)
    print(trainX.shape, trainY.shape)
    trainX = trainX.reshape((NIMAGES1-START)//TSTEPS, TSTEPS,
                            *trainX.shape[1:])
    trainY = trainY.reshape((NIMAGES1-START)//TSTEPS, TSTEPS,
                            *trainY.shape[1:])
    print(trainX.shape, trainY.shape)

    NIMAGES2 = 1000
    START = NIMAGES1

    valX, valY = load_image_for_keras(START=START, NUM_IMAGES=NIMAGES2)
    print(valX.shape, valY.shape)
    valX = valX.reshape((NIMAGES2-START)//TSTEPS, TSTEPS, *valX.shape[1:])
    valY = valY.reshape((NIMAGES2-START)//TSTEPS, TSTEPS, *valY.shape[1:])
    print(valX.shape, valY.shape)

    # XXX: Now build the keras model
    model = build_keras_model(trainX.shape)
    print(model.summary())

    # XXX: Now fit the model
    history = keras_model_fit(model, trainX, trainY, valX, valY)

    # XXX: Save the model after training
    model.save('model2.keras')

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
