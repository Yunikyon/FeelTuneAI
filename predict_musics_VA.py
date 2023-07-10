from __future__ import unicode_literals

import csv
import os
import numpy as np
import librosa
import joblib
import pandas as pd
import tensorflow as tf
from keras.api import keras


def predict_music_directory_emotions(directory, csv_name):
    # Get all files in the directory
    files = os.listdir(directory)

    # --- Extract Common features using librosa
    # Mel-frequency cepstral coefficients (MFCCs)
    # TODO - define fmin and fmax - but see first the average
    mfcc = np.array([librosa.feature.mfcc(y=librosa.load(directory + '/' + file, duration=150)[0],
                                          sr=librosa.load(directory + '/' + file, duration=150)[1])
                     for file in files])
    print(np.min(mfcc))  # -667.08264
    print(np.max(mfcc))  # 260.12265
    mfcc = np.around(np.interp(mfcc, (-700, 300), (0, 1)), decimals=3)
    print("Mfcc done!")

    # Spectral centroid
    # TODO - define fmin and fmax - but see first the average
    cent = np.array(
        [librosa.feature.spectral_centroid(y=librosa.load(directory + '/' + file, duration=150)[0]) for file in
         files])
    print(np.min(cent))  # 0.0
    print(np.max(cent))  # 8504.014129762603
    cent = np.around(np.interp(cent, (0, 8550), (0, 1)), decimals=3)
    print("cent done!")

    # TODO - Acho que Range: [0; 1]
    # TODO - normalizar para 3 casas decimais
    # Zero-crossing rate
    zcr = np.array(
        [librosa.feature.zero_crossing_rate(y=librosa.load(directory + '/' + file, duration=150)[0]) for file in
         files])
    print(np.min(zcr))  # 0.0
    print(np.max(zcr))  # 0.82666015625
    zcr = np.around(zcr, decimals=3)
    print("zcr done!")

    # --- Extract Valence features using librosa
    # Chroma features
    # TODO - Acho que Range: [0; 1]
    # TODO - normalizar para 3 casas decimais
    chroma_cqt = np.array(
        [librosa.feature.chroma_cqt(y=librosa.load(directory + '/' + file, duration=150)[0],
                                    sr=librosa.load(directory + '/' + file, duration=150)[1])
         for file in files])
    print(np.min(chroma_cqt))  # 0.0049088965
    print(np.max(chroma_cqt))  # 1.0
    chroma_cqt = np.around(chroma_cqt, decimals=3)
    print("Chroma_cqt done!")

    # Range: [0; 1]
    # TODO - normalizar para 3 casas decimais
    chroma_stft = np.array(
        [librosa.feature.chroma_stft(y=librosa.load(directory + '/' + file, duration=150)[0],
                                     sr=librosa.load(directory + '/' + file, duration=150)[1])
         for file in files])
    print(np.min(chroma_stft))  # 0.0
    print(np.max(chroma_stft))  # 1.0
    chroma_stft = np.around(chroma_stft, decimals=3)
    print("chroma_stft done!")

    # Range: [0; 1]
    # TODO - normalizar para 3 casas decimais
    chroma_cens = np.array(
        [librosa.feature.chroma_cens(y=librosa.load(directory + '/' + file, duration=150)[0],
                                     sr=librosa.load(directory + '/' + file, duration=150)[1])
         for file in files])
    print(np.min(chroma_cens))  # 0.0
    print(np.max(chroma_cens))  # 1.0
    chroma_cens = np.around(chroma_cens, decimals=3)
    print("chroma_cens done!")

    # Spectral rolloff
    # TODO - define min and max - but see first the average
    rolloff = np.array(
        [librosa.feature.spectral_rolloff(y=librosa.load(directory + '/' + file, duration=150)[0],
                                          sr=librosa.load(directory + '/' + file, duration=150)[1])
         for file in files])
    print(np.min(rolloff))  # 0.0
    print(np.max(rolloff))  # 10562.0361328125
    rolloff = np.around(np.interp(rolloff, (0, 10800), (0, 1)), decimals=3)
    print("rolloff done!")

    # --- Extract Arousal features using librosa
    # Spectral contrast
    # TODO - define fmin and fmax - but see first the average - I think it goes from 0 to 65
    spectral_contrast = np.array(
        [librosa.feature.spectral_contrast(y=librosa.load(directory + '/' + file, duration=150)[0],
                                           sr=librosa.load(directory + '/' + file, duration=150)[1])
         for file in files])
    print(np.min(spectral_contrast))  # 0.17194677838910621
    print(np.max(spectral_contrast))  # 72.256236009688
    spectral_contrast = np.around(np.interp(spectral_contrast, (0, 100), (0, 1)), decimals=3)
    print("Spectral_contrast done!")

    # Create individual DataFrames for each array
    dataframe_mfcc = pd.DataFrame(mfcc.reshape(mfcc.shape[0], -1))
    dataframe_cent = pd.DataFrame(cent.reshape(cent.shape[0], -1))
    dataframe_zcr = pd.DataFrame(zcr.reshape(zcr.shape[0], -1))
    dataframe_chroma_cqt = pd.DataFrame(chroma_cqt.reshape(chroma_cqt.shape[0], -1))
    dataframe_chroma_stft = pd.DataFrame(chroma_stft.reshape(chroma_stft.shape[0], -1))
    dataframe_chroma_cens = pd.DataFrame(chroma_cens.reshape(chroma_cens.shape[0], -1))
    dataframe_rolloff = pd.DataFrame(rolloff.reshape(rolloff.shape[0], -1))
    dataframe_spectral_contrast = pd.DataFrame(spectral_contrast.reshape(spectral_contrast.shape[0], -1))

    # Concatenate the DataFrames along the column axis (axis=1)
    dataframe_valence = pd.concat([dataframe_mfcc, dataframe_cent, dataframe_zcr, dataframe_chroma_cqt,
                                   dataframe_chroma_stft, dataframe_chroma_cens, dataframe_rolloff], axis=1)

    dataframe_arousal = pd.concat([dataframe_mfcc, dataframe_cent, dataframe_zcr, dataframe_spectral_contrast], axis=1)


    print("Predicting valence...")
    # Use the trained model to make predictions
    valence_model = keras.models.load_model(f'./models/valence_model.h5')
    predicted_valence = valence_model.predict(dataframe_valence)
    valence = np.around(np.interp(predicted_valence, (0, 1), (-1, 1)), decimals=3)

    print("Predicting arousal...")
    arousal_model = keras.models.load_model(f'./models/arousal_model.h5')
    predicted_arousal = arousal_model.predict(dataframe_arousal)
    arousal = np.around(np.interp(predicted_arousal, (0, 1), (-1, 1)), decimals=3)

    # Write to csv
    with open(f'{csv_name}.csv', 'w+', newline='', encoding="utf-8") as csv_file:
        header_row = ['music_name', 'music_valence', 'music_arousal']
        delimiter = '~~~'
        h = delimiter.join(header_row)
        csv_file.write(h + '\n')

        i = 0
        for music in files:
            l = delimiter.join([music, str(valence[i][0]), str(arousal[i][0])])
            csv_file.write(l + '\n')
            i += 1

def predict_uploaded_music_emotions(directory, file, csv_name):
    # Extract Common features using librosa
    # Mel-frequency cepstral coefficients (MFCCs)
    mfcc = np.array([librosa.feature.mfcc(y=librosa.load(directory + '/' + file, duration=150)[0], sr=44100)])
    print("Mfcc done!")
    # Spectral centroid
    cent = np.array(
        [librosa.feature.spectral_centroid(y=librosa.load(directory + '/' + file, duration=150)[0])])
    print("cent done!")
    # Zero-crossing rate
    zcr = np.array(
        [librosa.feature.zero_crossing_rate(y=librosa.load(directory + '/' + file, duration=150)[0])])
    print("zcr done!")

    # Extract Valence features using librosa
    # Chroma features
    chroma_cqt = np.array(
        [librosa.feature.chroma_cqt(y=librosa.load(directory + '/' + file, duration=150)[0], sr=44100)])
    print("Chroma_cqt done!")
    chroma_stft = np.array(
        [librosa.feature.chroma_stft(y=librosa.load(directory + '/' + file, duration=150)[0], sr=44100)])
    print("chroma_stft done!")
    chroma_cens = np.array(
        [librosa.feature.chroma_cens(y=librosa.load(directory + '/' + file, duration=150)[0], sr=44100)])
    print("chroma_cens done!")
    # Spectral rolloff
    rolloff = np.array(
        [librosa.feature.spectral_rolloff(y=librosa.load(directory + '/' + file, duration=150)[0], sr=44100)])
    print("rolloff done!")
    # rms
    rms = np.array(
        [librosa.feature.rms(y=librosa.load(directory + '/' + file, duration=150)[0])])
    print("rms done!")

    # Extract Arousal features using librosa
    # Spectral contrast
    spectral_contrast = np.array(
        [librosa.feature.spectral_contrast(y=librosa.load(directory + '/' + file, duration=150)[0], sr=44100)])
    print("Spectral_contrast done!")

    # Convert the features to NumPy arrays with the same shape
    # 1. Get the max length
    max_shape_valence = max(mfcc.shape[1], cent.shape[1], zcr.shape[1], rolloff.shape[1], chroma_cqt.shape[1],
                            chroma_stft.shape[1], chroma_cens.shape[1], rms.shape[1])
    max_shape_arousal = max(mfcc.shape[1], cent.shape[1], zcr.shape[1], spectral_contrast.shape[1])

    # 2.1 Convert Valence to the same shape
    if max_shape_valence - mfcc.shape[1] != 0:
        mfcc_valence = np.pad(mfcc, ((0, 0), (0, max_shape_valence - mfcc.shape[1]), (0, 0)), mode='constant',
                              constant_values=0)
    else:
        mfcc_valence = mfcc
    if max_shape_valence - cent.shape[1] != 0:
        cent_valence = np.pad(cent, ((0, 0), (0, max_shape_valence - cent.shape[1]), (0, 0)), mode='constant',
                              constant_values=0)
    else:
        cent_valence = cent
    if max_shape_valence - zcr.shape[1] != 0:
        zcr_valence = np.pad(zcr, ((0, 0), (0, max_shape_valence - zcr.shape[1]), (0, 0)), mode='constant',
                             constant_values=0)
    else:
        zcr_valence = zcr
    if max_shape_valence - rolloff.shape[1] != 0:
        rolloff = np.pad(rolloff, ((0, 0), (0, max_shape_valence - rolloff.shape[1]), (0, 0)), mode='constant',
                         constant_values=0)
    if max_shape_valence - chroma_cqt.shape[1] != 0:
        chroma_cqt = np.pad(chroma_cqt, ((0, 0), (0, max_shape_valence - chroma_cqt.shape[1]), (0, 0)), mode='constant',
                            constant_values=0)
    if max_shape_valence - chroma_stft.shape[1] != 0:
        chroma_stft = np.pad(chroma_stft, ((0, 0), (0, max_shape_valence - chroma_stft.shape[1]), (0, 0)),
                             mode='constant',
                             constant_values=0)
    if max_shape_valence - chroma_cens.shape[1] != 0:
        chroma_cens = np.pad(chroma_cens, ((0, 0), (0, max_shape_valence - chroma_cens.shape[1]), (0, 0)),
                             mode='constant',
                             constant_values=0)
    if max_shape_valence - rms.shape[1] != 0:
        rms = np.pad(rms, ((0, 0), (0, max_shape_valence - rms.shape[1]), (0, 0)),
                     mode='constant',
                     constant_values=0)

    # 2.2 Convert Arousal to the same shape
    if max_shape_arousal - mfcc.shape[1] != 0:
        mfcc_arousal = np.pad(mfcc, ((0, 0), (0, max_shape_arousal - mfcc.shape[1]), (0, 0)), mode='constant',
                              constant_values=0)
    else:
        mfcc_arousal = mfcc
    if max_shape_arousal - cent.shape[1] != 0:
        cent_arousal = np.pad(cent, ((0, 0), (0, max_shape_arousal - cent.shape[1]), (0, 0)), mode='constant',
                              constant_values=0)
    else:
        cent_arousal = cent
    if max_shape_arousal - zcr.shape[1] != 0:
        zcr_arousal = np.pad(zcr, ((0, 0), (0, max_shape_arousal - zcr.shape[1]), (0, 0)), mode='constant',
                             constant_values=0)
    else:
        zcr_arousal = zcr
    if max_shape_arousal - spectral_contrast.shape[1] != 0:
        spectral_contrast = np.pad(spectral_contrast,
                                   ((0, 0), (0, max_shape_arousal - spectral_contrast.shape[1]), (0, 0)),
                                   mode='constant',
                                   constant_values=0)

    print("Concatenating...")
    # Concatenate all features into one matrix for valence
    x_valence = np.concatenate(
        (mfcc_valence, cent_valence, zcr_valence, chroma_cqt, chroma_stft, chroma_cens, rolloff, rms), axis=2).reshape(1, -1)

    # Concatenate all features into one matrix for arousal
    x_arousal = np.concatenate((mfcc_arousal, cent_arousal, zcr_arousal, spectral_contrast), axis=2).reshape(1, -1)

    # Use the trained model to make predictions
    valence_model = joblib.load('./models/svm_valence_classifier.pkl')
    valence = valence_model.predict(x_valence)

    arousal_model = joblib.load('./models/svm_arousal_classifier.pkl')
    arousal = arousal_model.predict(x_arousal)

    # Write to csv
    with open(f'{csv_name}.csv', 'a', newline='', encoding="utf-8") as csv_file:
        delimiter = '~~~'
        l = delimiter.join([file, str(round(valence[0], 3)), str(round(arousal[0], 3))])
        csv_file.write(l + '\n')


if __name__ == '__main__':
    predict_music_directory_emotions('./BuildingDatasetPhaseMusics', 'building_dataset_phase_musics_va_2')

# predict_uploaded_music_emotions('./BuildingDatasetPhaseMusics', 'Avril Lavigne - Girlfriend (Official Video).mp3', 'building_dataset_phase_musics_va')

# predict_dataset_emotions('musics_to_classify', 'musics_classified')
