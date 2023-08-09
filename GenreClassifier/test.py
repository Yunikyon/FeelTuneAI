import os

import librosa
import numpy as np
import pandas as pd
from keras.layers import Flatten, Dense, BatchNormalization, Dropout
from numpy import var, mean, max, where
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import catboost as cb
from xgboost import XGBClassifier
import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *

def train_model():
    music_data = pd.read_csv('file.csv')
    # music_data.head(5)
    # print(music_data)
    # print(music_data['label'].value_counts())

    # Map label from classes to numbers (0 to 9)
    label_encoder = preprocessing.LabelEncoder()
    music_data['label'] = label_encoder.fit_transform(music_data['label'])

    # Define the variables
    X = music_data.drop(['label', 'filename'], axis=1)
    y = music_data['label']

    # Scale the data
    cols = X.columns
    minmax = preprocessing.MinMaxScaler()
    np_scaled = minmax.fit_transform(X)

    # new data frame with the new scaled data.
    X = pd.DataFrame(np_scaled, columns=cols)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=111)
    # X_train.shape, X_test.shape, y_train.shape, y_test.shape



    # rf = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)
    # cbc = cb.CatBoostClassifier(verbose=0, eval_metric='Accuracy', loss_function='MultiClass')
    # xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05)

    # for clf in (rf, cbc, xgb):
    #     clf.fit(X_train, y_train)
    #     preds = clf.predict(X_test)
    #     print(clf.__class__.__name__, accuracy_score(y_test, preds))
    #

    model = Sequential()

    model.add(Flatten(input_shape=(58,)))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='softmax'))

    # compile the model
    adam = keras.optimizers.Adam(lr=1e-4)
    model.compile(optimizer=adam,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    hist = model.fit(X_train, y_train,
                     validation_data=(X_test, y_test),
                     epochs=100,
                     batch_size=32)

    model_to_save = hist.model

    # Save the best model
    model_to_save.save("../models/genre_model.h5")

    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test accuracy: {test_accuracy}")

def extract_features(directory, csv_filename):
    # Get all files in the directory
    files = os.listdir(directory)

    # music_data = pd.read_csv('file.csv')

    data = []

    for file in files:
        # audio = librosa.load(directory + '/' + file, duration=30.0133)
        audio = librosa.load(directory + '/' + file, duration=150)

        # Length
        length = audio[0].shape[0]

        # Chroma_stft
        chroma_stft = np.array(librosa.feature.chroma_stft(y=audio[0], sr=audio[1]))
        chroma_stft_mean = mean(chroma_stft)
        chroma_stft_var = var(chroma_stft)

        # rms
        rms = np.array(librosa.feature.rms(y=audio[0]))
        rms_mean = mean(rms)
        rms_var = var(rms)

        # Spectral centroid
        spectral_centroid = np.array(librosa.feature.spectral_centroid(y=audio[0]))
        spectral_centroid_mean = mean(spectral_centroid)
        spectral_centroid_var = var(spectral_centroid)

        # Spectral bandwidth
        spectral_bandwidth = np.array(librosa.feature.spectral_bandwidth(y=audio[0], sr=audio[1]))
        spectral_bandwidth_mean = mean(spectral_bandwidth)
        spectral_bandwidth_var = var(spectral_bandwidth)

        # Rolloff
        rolloff = np.array(librosa.feature.spectral_rolloff(y=audio[0], sr=audio[1]))
        rolloff_mean = mean(rolloff)
        rolloff_var = var(rolloff)

        # Zero crossing rate
        zcr = np.array(librosa.feature.zero_crossing_rate(y=audio[0]))
        zcr_mean = mean(zcr)
        zcr_var = var(zcr)

        # harmony & perceptr
        harmony, perceptr = librosa.effects.hpss(audio[0])

        harmony_mean = mean(harmony)
        harmony_var = var(harmony)

        perceptr_mean = mean(perceptr)
        perceptr_var = mean(perceptr)

        # tempo
        tempo = np.array(librosa.feature.tempo(y=audio[0], sr=audio[1]))[0]

        # mfcc1... mfcc20
        mfccs = np.array(librosa.feature.mfcc(y=audio[0], sr=audio[1]))

        mfccs_arrays = np.split(mfccs, 20)

        mfcc1_mean = mean(mfccs_arrays[0])
        mfcc1_var = var(mfccs_arrays[0])

        mfcc2_mean = mean(mfccs_arrays[1])
        mfcc2_var = var(mfccs_arrays[1])

        mfcc3_mean = mean(mfccs_arrays[2])
        mfcc3_var = var(mfccs_arrays[2])

        mfcc4_mean = mean(mfccs_arrays[3])
        mfcc4_var = var(mfccs_arrays[3])

        mfcc5_mean = mean(mfccs_arrays[4])
        mfcc5_var = var(mfccs_arrays[4])

        mfcc6_mean = mean(mfccs_arrays[5])
        mfcc6_var = var(mfccs_arrays[5])

        mfcc7_mean = mean(mfccs_arrays[6])
        mfcc7_var = var(mfccs_arrays[6])

        mfcc8_mean = mean(mfccs_arrays[7])
        mfcc8_var = var(mfccs_arrays[7])

        mfcc9_mean = mean(mfccs_arrays[8])
        mfcc9_var = var(mfccs_arrays[8])

        mfcc10_mean = mean(mfccs_arrays[9])
        mfcc10_var = var(mfccs_arrays[9])

        mfcc11_mean = mean(mfccs_arrays[10])
        mfcc11_var = var(mfccs_arrays[10])

        mfcc12_mean = mean(mfccs_arrays[11])
        mfcc12_var = var(mfccs_arrays[11])

        mfcc13_mean = mean(mfccs_arrays[12])
        mfcc13_var = var(mfccs_arrays[12])

        mfcc14_mean = mean(mfccs_arrays[13])
        mfcc14_var = var(mfccs_arrays[13])

        mfcc15_mean = mean(mfccs_arrays[14])
        mfcc15_var = var(mfccs_arrays[14])

        mfcc16_mean = mean(mfccs_arrays[15])
        mfcc16_var = var(mfccs_arrays[15])

        mfcc17_mean = mean(mfccs_arrays[16])
        mfcc17_var = var(mfccs_arrays[16])

        mfcc18_mean = mean(mfccs_arrays[17])
        mfcc18_var = var(mfccs_arrays[17])

        mfcc19_mean = mean(mfccs_arrays[18])
        mfcc19_var = var(mfccs_arrays[18])

        mfcc20_mean = mean(mfccs_arrays[19])
        mfcc20_var = var(mfccs_arrays[19])

        data.append([file, length, chroma_stft_mean, chroma_stft_var, rms_mean, rms_var, spectral_centroid_mean,
                     spectral_centroid_var, spectral_bandwidth_mean, spectral_bandwidth_var, rolloff_mean, rolloff_var,
                     zcr_mean, zcr_var, harmony_mean, harmony_var, perceptr_mean, perceptr_var, tempo, mfcc1_mean,
                     mfcc1_var, mfcc2_mean, mfcc2_var, mfcc3_mean, mfcc3_var, mfcc4_mean, mfcc4_var, mfcc5_mean,
                     mfcc5_var, mfcc6_mean, mfcc6_var, mfcc7_mean, mfcc7_var, mfcc8_mean, mfcc8_var, mfcc9_mean,
                     mfcc9_var, mfcc10_mean, mfcc10_var, mfcc11_mean, mfcc11_var, mfcc12_mean, mfcc12_var, mfcc13_mean,
                     mfcc13_var, mfcc14_mean, mfcc14_var, mfcc15_mean, mfcc15_var, mfcc16_mean, mfcc16_var, mfcc17_mean,
                     mfcc17_var, mfcc18_mean, mfcc18_var, mfcc19_mean, mfcc19_var, mfcc20_mean, mfcc20_var])

        print(f'Music {file}\'s features extracted!')


    # --- Extract features using librosa
    # Mel-frequency cepstral coefficients (MFCCs)
    # mfcc = np.array([librosa.feature.mfcc(y=librosa.load(directory + '/' + file, duration=30)[0],
    #                                       sr=librosa.load(directory + '/' + file, duration=30)[1])
    #                  for file in files])
    # zcr = np.array(librosa.feature.zero_crossing_rate(y=audio_one[0]))

    # --- Save file with the features

    # Column names
    columns = [
        "filename", "length", "chroma_sftf_mean", "chroma_sftf_var", "rms_mean", "rms_var",
        "spectral_centroid_mean", "spectral_centroid_var", "spectral_bandwidth_mean", "spectral_bandwidth_var",
        "rolloff_mean", "rolloff_var", "zero_crossing_rate_mean", "zero_crossing_rate_var",
        "harmony_mean", "harmony_var", "perceptr_mean", "perceptr_var", "tempo",
        "mfcc1_mean", "mfcc1_var", "mfcc2_mean", "mfcc2_var", "mfcc3_mean", "mfcc3_var",
        "mfcc4_mean", "mfcc4_var", "mfcc5_mean", "mfcc5_var", "mfcc6_mean", "mfcc6_var",
        "mfcc7_mean", "mfcc7_var", "mfcc8_mean", "mfcc8_var", "mfcc9_mean", "mfcc9_var",
        "mfcc10_mean", "mfcc10_var", "mfcc11_mean", "mfcc11_var", "mfcc12_mean", "mfcc12_var",
        "mfcc13_mean", "mfcc13_var", "mfcc14_mean", "mfcc14_var", "mfcc15_mean", "mfcc15_var",
        "mfcc16_mean", "mfcc16_var", "mfcc17_mean", "mfcc17_var", "mfcc18_mean", "mfcc18_var",
        "mfcc19_mean", "mfcc19_var", "mfcc20_mean", "mfcc20_var"
    ]

    # Create a DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Save the DataFrame to a CSV file
    df.to_csv(csv_filename, index=False)


def predict_musics_genres(directory, csv_filename_features, csv_filename_genres):
    extract_features(directory, csv_filename_features)
    music_data = pd.read_csv(csv_filename_features)

    # Define the variables
    X = music_data.drop(['filename'], axis=1)

    # Scale the data
    cols = X.columns
    minmax = preprocessing.MinMaxScaler()
    np_scaled = minmax.fit_transform(X)

    # new data frame with the new scaled data.
    X = pd.DataFrame(np_scaled, columns=cols)

    # Load model and predict
    genre_model = keras.models.load_model(f'../models/genre_model.h5')
    predicted_genres = genre_model.predict(X)

    # Column names
    columns = [
        "filename", "genre"
    ]

    files = os.listdir(directory)

    data = []

    genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

    i = 0
    for file in files:
        index = where(predicted_genres[i] == max(predicted_genres[i]))[0][0]
        i = i+1

        data.append([file, genres[index]])

    # Create a DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Save the DataFrame to a CSV file
    df.to_csv(csv_filename_genres, index=False)

if __name__ == '__main__':
    # train_model()
    # extract_features('../musics', '../musics_genre_features.csv')
    # predict_musics_genres('../musics', '../musics_genre_features.csv', '../musics_genre_predicted.csv')
    predict_musics_genres('../musics', '../musics_genre_features_2.csv', '../musics_genre_predicted_2.csv')