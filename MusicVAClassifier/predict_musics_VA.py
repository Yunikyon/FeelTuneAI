from __future__ import unicode_literals

import csv
import os
import sqlite3

import numpy as np
import librosa
import joblib
import pandas as pd
import tensorflow as tf
from keras.api import keras

from download_from_yt import download_musics_from_csv


def predict_music_directory_va(directory):
    # Get all files in the directory
    files = os.listdir(directory)

    # --- Extract Common features using librosa
    # Mel-frequency cepstral coefficients (MFCCs)
    mfcc = np.array([librosa.feature.mfcc(y=librosa.load(directory + '/' + file, duration=150)[0],
                                          sr=librosa.load(directory + '/' + file, duration=150)[1])
                     for file in files])
    print(np.min(mfcc))  # -667.08264
    print(np.max(mfcc))  # 260.12265
    mfcc = np.around(np.interp(mfcc, (-700, 300), (0, 1)), decimals=3)
    print("Mfcc done!")

    # Spectral centroid
    cent = np.array(
        [librosa.feature.spectral_centroid(y=librosa.load(directory + '/' + file, duration=150)[0]) for file in
         files])
    print(np.min(cent))  # 0.0
    print(np.max(cent))  # 8504.014129762603
    cent = np.around(np.interp(cent, (0, 8550), (0, 1)), decimals=3)
    print("cent done!")

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
    chroma_cqt = np.array(
        [librosa.feature.chroma_cqt(y=librosa.load(directory + '/' + file, duration=150)[0],
                                    sr=librosa.load(directory + '/' + file, duration=150)[1])
         for file in files])
    print(np.min(chroma_cqt))  # 0.0049088965
    print(np.max(chroma_cqt))  # 1.0
    chroma_cqt = np.around(chroma_cqt, decimals=3)
    print("Chroma_cqt done!")

    chroma_stft = np.array(
        [librosa.feature.chroma_stft(y=librosa.load(directory + '/' + file, duration=150)[0],
                                     sr=librosa.load(directory + '/' + file, duration=150)[1])
         for file in files])
    print(np.min(chroma_stft))  # 0.0
    print(np.max(chroma_stft))  # 1.0
    chroma_stft = np.around(chroma_stft, decimals=3)
    print("chroma_stft done!")

    chroma_cens = np.array(
        [librosa.feature.chroma_cens(y=librosa.load(directory + '/' + file, duration=150)[0],
                                     sr=librosa.load(directory + '/' + file, duration=150)[1])
         for file in files])
    print(np.min(chroma_cens))  # 0.0
    print(np.max(chroma_cens))  # 1.0
    chroma_cens = np.around(chroma_cens, decimals=3)
    print("chroma_cens done!")

    # Spectral rolloff
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
    valence_model = keras.models.load_model(f'Models/valence_model.h5')
    predicted_valence = valence_model.predict(dataframe_valence)
    valence = np.around(np.interp(predicted_valence, (0, 1), (-1, 1)), decimals=3)

    print("Predicting arousal...")
    arousal_model = keras.models.load_model(f'Models/arousal_model.h5')
    predicted_arousal = arousal_model.predict(dataframe_arousal)
    arousal = np.around(np.interp(predicted_arousal, (0, 1), (-1, 1)), decimals=3)

    # Save in database
    conn = sqlite3.connect('../Database/feeltune.db')
    cursor = conn.cursor()
    i = 0
    for music in files:
        cursor.execute("INSERT INTO musics (name, valence, arousal) VALUES (?, ?, ?)",
                       (music, valence[i][0], arousal[i][0]))
        music_id = cursor.lastrowid
        cursor.execute("INSERT INTO user_musics (user_id, music_id) VALUES (?, ?)", (0, music_id))
        i += 1

    conn.commit()
    conn.close()


def predict_uploaded_music_va(directory, file, user_uploaded=None):
    # --- Extract Common features using librosa
    # Mel-frequency cepstral coefficients (MFCCs)
    mfcc = np.array([librosa.feature.mfcc(y=librosa.load(directory + '/' + file, duration=150)[0],
                                          sr=librosa.load(directory + '/' + file, duration=150)[1])])
    print(np.min(mfcc))  # -667.08264
    print(np.max(mfcc))  # 260.12265
    mfcc = np.around(np.interp(mfcc, (-700, 300), (0, 1)), decimals=3)
    print("Mfcc done!")

    # Spectral centroid
    cent = np.array(
        [librosa.feature.spectral_centroid(y=librosa.load(directory + '/' + file, duration=150)[0])])
    print(np.min(cent))  # 0.0
    print(np.max(cent))  # 8504.014129762603
    cent = np.around(np.interp(cent, (0, 8550), (0, 1)), decimals=3)
    print("cent done!")

    # Zero-crossing rate
    zcr = np.array(
        [librosa.feature.zero_crossing_rate(y=librosa.load(directory + '/' + file, duration=150)[0])])
    print(np.min(zcr))  # 0.0
    print(np.max(zcr))  # 0.82666015625
    zcr = np.around(zcr, decimals=3)
    print("zcr done!")

    # --- Extract Valence features using librosa
    # Chroma features
    chroma_cqt = np.array(
        [librosa.feature.chroma_cqt(y=librosa.load(directory + '/' + file, duration=150)[0],
                                    sr=librosa.load(directory + '/' + file, duration=150)[1])])
    print(np.min(chroma_cqt))  # 0.0049088965
    print(np.max(chroma_cqt))  # 1.0
    chroma_cqt = np.around(chroma_cqt, decimals=3)
    print("Chroma_cqt done!")

    chroma_stft = np.array(
        [librosa.feature.chroma_stft(y=librosa.load(directory + '/' + file, duration=150)[0],
                                     sr=librosa.load(directory + '/' + file, duration=150)[1])])
    print(np.min(chroma_stft))  # 0.0
    print(np.max(chroma_stft))  # 1.0
    chroma_stft = np.around(chroma_stft, decimals=3)
    print("chroma_stft done!")

    chroma_cens = np.array(
        [librosa.feature.chroma_cens(y=librosa.load(directory + '/' + file, duration=150)[0],
                                     sr=librosa.load(directory + '/' + file, duration=150)[1])])
    print(np.min(chroma_cens))  # 0.0
    print(np.max(chroma_cens))  # 1.0
    chroma_cens = np.around(chroma_cens, decimals=3)
    print("chroma_cens done!")

    # Spectral rolloff
    rolloff = np.array(
        [librosa.feature.spectral_rolloff(y=librosa.load(directory + '/' + file, duration=150)[0],
                                          sr=librosa.load(directory + '/' + file, duration=150)[1])])
    print(np.min(rolloff))  # 0.0
    print(np.max(rolloff))  # 10562.0361328125
    rolloff = np.around(np.interp(rolloff, (0, 10800), (0, 1)), decimals=3)
    print("rolloff done!")

    # --- Extract Arousal features using librosa
    # Spectral contrast
    spectral_contrast = np.array(
        [librosa.feature.spectral_contrast(y=librosa.load(directory + '/' + file, duration=150)[0],
                                           sr=librosa.load(directory + '/' + file, duration=150)[1])])
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
    valence_model = keras.models.load_model(f'../Models/valence_model.h5')
    predicted_valence = valence_model.predict(dataframe_valence)
    valence = np.around(np.interp(predicted_valence, (0, 1), (-1, 1)), decimals=3)

    print("Predicting arousal...")
    arousal_model = keras.models.load_model(f'../Models/arousal_model.h5')
    predicted_arousal = arousal_model.predict(dataframe_arousal)
    arousal = np.around(np.interp(predicted_arousal, (0, 1), (-1, 1)), decimals=3)

    conn = sqlite3.connect('../Database/feeltune.db')
    cursor = conn.cursor()

    cursor.execute("INSERT INTO musics (name, valence, arousal) VALUES (?, ?, ?)",
                   (file, valence[0][0], arousal[0][0]))
    music_id = cursor.lastrowid
    if user_uploaded is None:
        user_id = 0
    else:
        cursor.execute("SELECT id FROM users WHERE name = ?", (user_uploaded,))
        user_id = cursor.fetchone()[0]
    cursor.execute("INSERT INTO user_musics (user_id, music_id) VALUES (?, ?)", (user_id, music_id))

    conn.commit()
    conn.close()


if __name__ == '__main__':
    if not os.path.exists('Musics'):
        os.makedirs('Musics')

    download_musics_from_csv('../bdp_musics_id.csv', '../Musics')
    predict_music_directory_va('../Musics')
