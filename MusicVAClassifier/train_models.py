from __future__ import unicode_literals
import csv
import youtube_dl
import numpy as np
import pandas as pd
import librosa
from sklearn.svm import SVR, NuSVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib


class Bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def download_mu_vi_musics():
    class MyLogger(object):
        def debug(self, msg):
            pass

        def warning(self, msg):
            pass

        def error(self, msg):
            print(msg)

    def my_hook(d):
        if d['status'] == 'finished':
            print('Done downloading, now converting ...')

    def download_music(youtube_id):
        save_path = 'MuVi_musics'

        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'logger': MyLogger(),
            'progress_hooks': [my_hook],
            'outtmpl': save_path + '/%(title)s.%(ext)s'
        }

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            url = 'https://www.youtube.com/watch?v=' + youtube_id
            info_dict = ydl.extract_info(url, download=False)
            video_title = info_dict.get('title', None) + '.mp3'
            print(video_title)
            ydl.download([url])
            return video_title

    def download_from_csv_with_yt_ids(va_dataset_csv):
        music_index = 0

        # Create csv file to save music name with corresponding id
        with open('musics_and_ids.csv', 'w+', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            header = ['Music_name', 'Music_id']
            writer.writerow(header)

        # Files with id of the music url of YouTube
        with open(va_dataset_csv) as file_obj:
            reader_obj = csv.reader(file_obj)

            # Skips the heading
            next(file_obj)

            # Download musics from YouTube with youtube-dl
            for row in reader_obj:
                music_id = row[0].split(';')[0]
                print("New music - " + music_id)

                try:
                    music_name = download_music(music_id)

                    # Save music name and id - for better understanding
                    with open('musics_and_ids.csv', 'a', encoding='UTF8', newline='') as f:
                        writer = csv.writer(f)
                        row = [music_name, music_id]
                        # Add new music file
                        writer.writerow(row)

                    print("Music downloaded")
                    music_index += 1
                except Exception as e:
                    print(f"{Bcolors.WARNING} Error on music id - " + music_id + Bcolors.ENDC)

    download_from_csv_with_yt_ids('va_dataset.csv')


def build_model():
    # Load arousal_valence dataset
    dataset = pd.read_csv("training_dataset.csv")
    print("Dataset read")

    # Extract Common features using librosa
    # Mel-frequency cepstral coefficients (MFCCs)
    mfcc = np.array([librosa.feature.mfcc(y=librosa.load('MuVi_musics/' + file, duration=150)[0], sr=44100) for file in
                     dataset["Music_name"]])
    print("Mfcc done!")
    # Spectral centroid
    cent = np.array(
        [librosa.feature.spectral_centroid(y=librosa.load('MuVi_musics/' + file, duration=150)[0]) for file in
         dataset["Music_name"]])
    print("cent done!")
    # Zero-crossing rate
    zcr = np.array(
        [librosa.feature.zero_crossing_rate(y=librosa.load('MuVi_musics/' + file, duration=150)[0]) for file in
         dataset["Music_name"]])
    print("zcr done!")

    # Extract Valence features using librosa
    # Chroma features
    chroma_cqt = np.array(
        [librosa.feature.chroma_cqt(y=librosa.load('MuVi_musics/' + file, duration=150)[0], sr=44100) for file in
         dataset["Music_name"]])
    print("Chroma_cqt done!")
    chroma_stft = np.array(
        [librosa.feature.chroma_stft(y=librosa.load('MuVi_musics/' + file, duration=150)[0], sr=44100) for file in
         dataset["Music_name"]])
    print("chroma_stft done!")
    chroma_cens = np.array(
        [librosa.feature.chroma_cens(y=librosa.load('MuVi_musics/' + file, duration=150)[0], sr=44100) for file in
         dataset["Music_name"]])
    print("chroma_cens done!")
    # Spectral rolloff
    rolloff = np.array(
        [librosa.feature.spectral_rolloff(y=librosa.load('MuVi_musics/' + file, duration=150)[0], sr=44100) for file in
         dataset["Music_name"]])
    print("rolloff done!")
    # rms
    rms = np.array(
        [librosa.feature.rms(y=librosa.load('MuVi_musics/' + file, duration=150)[0]) for file in
         dataset["Music_name"]])
    print("rms done!")

    # Extract Arousal features using librosa
    # Spectral contrast
    spectral_contrast = np.array(
        [librosa.feature.spectral_contrast(y=librosa.load('MuVi_musics/' + file, duration=150)[0], sr=44100) for file in
         dataset["Music_name"]])
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
        len(dataset), -1)
    y_valence = dataset["Valence"].values

    # Concatenate all features into one matrix for arousal
    x_arousal = np.concatenate((mfcc_arousal, cent_arousal, zcr_arousal, spectral_contrast), axis=2).reshape(
        len(dataset), -1)
    y_arousal = dataset["Arousal"].values

    print("Train and test split...")
    # Split data into training and testing sets for Valence
    x_train_valence, x_test_valence, y_train_valence, y_test_valence = train_test_split(x_valence, y_valence,
                                                                                        test_size=0.2, random_state=42)
    x_train_arousal, x_test_arousal, y_train_arousal, y_test_arousal = train_test_split(x_arousal, y_arousal,
                                                                                        test_size=0.2, random_state=42)

    print("Training...")

    # Train SVM classifier for valence
    svm_valence = NuSVR(kernel='rbf', nu=0.5, gamma='scale')  # Valence score = 0.3364867410372002
    svm_valence.fit(x_train_valence, y_train_valence)
    # svm_valence = SVR(kernel='rbf', C=1, gamma='scale') # Valence score = 0.32006080875792065
    # svm_valence = RandomForestRegressor(n_estimators=100, random_state=42) # Valence score = 0.18499682846047716
    # svm_valence = SVR(kernel='linear') # Valence score = 0.2957333921408458

    # Evaluate valence model
    y_pred_valence = svm_valence.predict(x_test_valence)
    mse = mean_squared_error(y_test_valence, y_pred_valence)
    print("Valence MSE: ", mse)
    print("Valence RMSE: ", mse * (1 / 2.0))

    # Save valence model
    model_file = "../models/svm_valence_classifier.pkl"
    joblib.dump(svm_valence, model_file)

    # Train SVM classifier for arousal
    svm_arousal = SVR(kernel='linear', C=1)  # Arousal score = 0.18566561362623968
    svm_arousal.fit(x_train_arousal, y_train_arousal)
    # svm_arousal = SVR(kernel='rbf', C=1, gamma='scale') # Arousal score = 0.1672412704940316
    # svm_arousal = RandomForestRegressor(n_estimators=100, random_state=42) # Arousal score = -0.1780557406683747
    # svm_arousal = NuSVR(kernel='rbf', nu=0.5, gamma='scale') # Arousal score = 0.17019876993624228

    # Evaluate arousal model
    y_pred_arousal = svm_arousal.predict(x_test_arousal)
    mse = mean_squared_error(y_test_arousal, y_pred_arousal)
    print("Arousal MSE: ", mse)
    print("Arousal RMSE: ", mse * (1 / 2.0))

    # Save arousal model
    model_file = "../models/svm_arousal_classifier.pkl"
    joblib.dump(svm_arousal, model_file)


# download_mu_vi_musics()
build_model()

