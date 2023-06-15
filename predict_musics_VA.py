from __future__ import unicode_literals

import csv
import os
import numpy as np
import librosa
import joblib


def predict_music_directory_emotions(directory, csv_name):
    # Get all files in the directory
    files = os.listdir(directory)

    # Extract Common features using librosa
    # Mel-frequency cepstral coefficients (MFCCs)
    mfcc = np.array([librosa.feature.mfcc(y=librosa.load(directory + '/' + file, duration=150)[0], sr=44100) for file in
                     files], dtype=object)
    print("Mfcc done!")
    # Spectral centroid
    cent = np.array(
        [librosa.feature.spectral_centroid(y=librosa.load(directory + '/' + file, duration=150)[0]) for file in
         files], dtype=object)
    print("cent done!")
    # Zero-crossing rate
    zcr = np.array(
        [librosa.feature.zero_crossing_rate(y=librosa.load(directory + '/' + file, duration=150)[0]) for file in
         files], dtype=object)
    print("zcr done!")

    # Extract Valence features using librosa
    # Chroma features
    chroma_cqt = np.array(
        [librosa.feature.chroma_cqt(y=librosa.load(directory + '/' + file, duration=150)[0], sr=44100) for file in
         files], dtype=object)
    print("Chroma_cqt done!")
    chroma_stft = np.array(
        [librosa.feature.chroma_stft(y=librosa.load(directory + '/' + file, duration=150)[0], sr=44100) for file in
         files], dtype=object)
    print("chroma_stft done!")
    chroma_cens = np.array(
        [librosa.feature.chroma_cens(y=librosa.load(directory + '/' + file, duration=150)[0], sr=44100) for file in
         files], dtype=object)
    print("chroma_cens done!")
    # Spectral rolloff
    rolloff = np.array(
        [librosa.feature.spectral_rolloff(y=librosa.load(directory + '/' + file, duration=150)[0], sr=44100) for file in
         files], dtype=object)
    print("rolloff done!")
    # rms
    rms = np.array(
        [librosa.feature.rms(y=librosa.load(directory + '/' + file, duration=150)[0]) for file in
         files], dtype=object)
    print("rms done!")

    # Extract Arousal features using librosa
    # Spectral contrast
    spectral_contrast = np.array(
        [librosa.feature.spectral_contrast(y=librosa.load(directory + '/' + file, duration=150)[0], sr=44100) for file
         in files], dtype=object)
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
        (mfcc_valence, cent_valence, zcr_valence, chroma_cqt, chroma_stft, chroma_cens, rolloff, rms), axis=2).reshape(
        len(files), -1)

    # Concatenate all features into one matrix for arousal
    x_arousal = np.concatenate((mfcc_arousal, cent_arousal, zcr_arousal, spectral_contrast), axis=2).reshape(
        len(files), -1)

    # Use the trained model to make predictions
    valence_model = joblib.load('../models/svm_valence_classifier.pkl')
    valence = valence_model.predict(x_valence)

    arousal_model = joblib.load('../models/svm_arousal_classifier.pkl')
    arousal = arousal_model.predict(x_arousal)

    # Write to csv
    with open(f'{csv_name}.csv', 'w+', newline='', encoding="utf-8") as csv_file:
        header_row = ['music_name', 'music_valence', 'music_arousal']
        delimiter = '~~~'
        h = delimiter.join(header_row)
        csv_file.write(h + '\n')

        i = 0
        for music in files:
            l = delimiter.join([music, str(round(valence[i], 3)), str(round(arousal[i], 3))])
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
    valence_model = joblib.load('../models/svm_valence_classifier.pkl')
    valence = valence_model.predict(x_valence)

    arousal_model = joblib.load('../models/svm_arousal_classifier.pkl')
    arousal = arousal_model.predict(x_arousal)

    # Write to csv
    with open(f'{csv_name}.csv', 'a', newline='', encoding="utf-8") as csv_file:
        delimiter = '~~~'
        l = delimiter.join([file, str(round(valence[0], 3)), str(round(arousal[0], 3))])
        csv_file.write(l + '\n')


# predict_uploaded_music_emotions('./BuildingDatasetPhaseMusics', 'Avril Lavigne - Girlfriend (Official Video).mp3', 'building_dataset_phase_musics_va')

# predict_dataset_emotions('musics_to_classify', 'musics_classified')
