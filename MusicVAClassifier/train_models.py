from __future__ import unicode_literals
import csv
import os

import matplotlib.pyplot as plt
import optuna
import youtube_dl
import numpy as np
import pandas as pd
import librosa
from keras import Model
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Dropout
import optuna.visualization as vis
from sklearn import metrics


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

def build_models():
    # Load arousal_valence dataset
    dataset = pd.read_csv("training_dataset.csv")
    print("Dataset read")

    # --- Extract Common features using librosa
    # Mel-frequency cepstral coefficients (MFCCs)
    mfcc = np.array([librosa.feature.mfcc(y=librosa.load('MuVi_musics/' + file, duration=150)[0],
                                          sr=librosa.load('MuVi_musics/' + file, duration=150)[1])
                     for file in dataset["Music_name"]])
    print(np.min(mfcc))  # -667.08264
    print(np.max(mfcc))  # 260.12265
    mfcc = np.around(np.interp(mfcc, (-700, 300), (0, 1)), decimals=3)
    print("Mfcc done!")

    # Spectral centroid
    cent = np.array(
        [librosa.feature.spectral_centroid(y=librosa.load('MuVi_musics/' + file, duration=150)[0]) for file in
         dataset["Music_name"]])
    print(np.min(cent))  # 0.0
    print(np.max(cent))  # 8504.014129762603
    cent = np.around(np.interp(cent, (0, 8550), (0, 1)), decimals=3)
    print("cent done!")

    # Zero-crossing rate
    zcr = np.array(
        [librosa.feature.zero_crossing_rate(y=librosa.load('MuVi_musics/' + file, duration=150)[0]) for file in
         dataset["Music_name"]])
    print(np.min(zcr))  # 0.0
    print(np.max(zcr))  # 0.82666015625
    zcr = np.around(zcr, decimals=3)
    print("zcr done!")


    # --- Extract Valence features using librosa
    # Chroma features
    chroma_cqt = np.array(
        [librosa.feature.chroma_cqt(y=librosa.load('MuVi_musics/' + file, duration=150)[0],
                                    sr=librosa.load('MuVi_musics/' + file, duration=150)[1])
         for file in dataset["Music_name"]])
    print(np.min(chroma_cqt))  # 0.0049088965
    print(np.max(chroma_cqt))  # 1.0
    chroma_cqt = np.around(chroma_cqt, decimals=3)
    print("Chroma_cqt done!")

    chroma_stft = np.array(
        [librosa.feature.chroma_stft(y=librosa.load('MuVi_musics/' + file, duration=150)[0],
                                     sr=librosa.load('MuVi_musics/' + file, duration=150)[1])
         for file in dataset["Music_name"]])
    print(np.min(chroma_stft))  # 0.0
    print(np.max(chroma_stft))  # 1.0
    chroma_stft = np.around(chroma_stft, decimals=3)
    print("chroma_stft done!")

    chroma_cens = np.array(
        [librosa.feature.chroma_cens(y=librosa.load('MuVi_musics/' + file, duration=150)[0],
                                     sr=librosa.load('MuVi_musics/' + file, duration=150)[1])
         for file in dataset["Music_name"]])
    print(np.min(chroma_cens))  # 0.0
    print(np.max(chroma_cens))  # 1.0
    chroma_cens = np.around(chroma_cens, decimals=3)
    print("chroma_cens done!")

    # Spectral rolloff
    rolloff = np.array(
        [librosa.feature.spectral_rolloff(y=librosa.load('MuVi_musics/' + file, duration=150)[0],
                                          sr=librosa.load('MuVi_musics/' + file, duration=150)[1])
         for file in dataset["Music_name"]])
    print(np.min(rolloff))  # 0.0
    print(np.max(rolloff))  # 10562.0361328125
    rolloff = np.around(np.interp(rolloff, (0, 10800), (0, 1)), decimals=3)
    print("rolloff done!")

    # --- Extract Arousal features using librosa
    # Spectral contrast
    spectral_contrast = np.array(
        [librosa.feature.spectral_contrast(y=librosa.load('MuVi_musics/' + file, duration=150)[0],
                                           sr=librosa.load('MuVi_musics/' + file, duration=150)[1])
         for file in dataset["Music_name"]])
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

    print("Concatenating...")
    y_valence = np.around(np.interp(dataset["Valence"].values, (-1, 1), (0, 1)), decimals=3)
    y_arousal = np.around(np.interp(dataset["Arousal"].values, (-1, 1), (0, 1)), decimals=3)

    print("Train and test split...")
    # Split data into training and testing sets for Valence
    x_train_valence, x_test_valence, y_train_valence, y_test_valence = train_test_split(dataframe_valence, y_valence,
                                                                                        test_size=0.25, random_state=42)
    x_train_arousal, x_test_arousal, y_train_arousal, y_test_arousal = train_test_split(dataframe_arousal, y_arousal,
                                                                                        test_size=0.25, random_state=42)

    # Split data into training and validation sets for Valence
    x_train_valence, x_val_valence, y_train_valence, y_val_valence = train_test_split(x_train_valence, y_train_valence,
                                                                                        test_size=0.25, random_state=42)
    x_train_arousal, x_val_arousal, y_train_arousal, y_val_arousal = train_test_split(x_train_arousal, y_train_arousal,
                                                                                        test_size=0.25, random_state=42)

    # --- Valence Model With Optuna
    print(f"Training Valence...")
    study = optuna.create_study(direction='minimize')  # or 'maximize' if optimizing accuracy
    study.optimize(lambda trial: objective(trial, x_train_valence, x_test_valence, x_val_valence,
                                           y_train_valence, y_test_valence, y_val_arousal,
                                           'valence'), n_trials=150)

    # Plot and save the optimization history
    fig = vis.plot_optimization_history(study)
    fig.update_layout(title="Valence Optimization History", yaxis_title="MAPE")
    fig.write_image("./Optuna_History_images/valence_optuna_history.png")

    # Plot and save the slice plot
    fig = vis.plot_slice(study)
    fig.update_layout(title="Valence Slice Plot", yaxis_title="MAPE")
    fig.write_image("./Optuna_History_images/valence_optuna_slice_plot.png")

    best_trial = study.best_trial
    best_learning_rate = best_trial.params['learning_rate']
    best_num_units = best_trial.params['num_units']
    best_dropout_rate = best_trial.params['dropout_rate']
    best_epochs = best_trial.params['epochs']
    best_batch_size = best_trial.params['batch_size']

    best_metric_valence, best_model_valence = train_model(x_train_valence, x_test_valence, x_val_valence,
                                                          y_train_valence, y_test_valence, y_val_valence, 'valence',
                                                          best_learning_rate, best_num_units, best_dropout_rate,
                                                          best_epochs, best_batch_size)

    print('Best validation MAPE value: {:.5f}'.format(best_metric_valence))
    print('Best parameters: {}'.format(best_trial.params))

    # Save the best model
    best_model_valence.save("../Models/valence_model.h5")

    # --- Arousal Model With Optuna
    print(f"Training Arousal...")
    study_arousal = optuna.create_study(direction='minimize')  # or 'maximize' if optimizing accuracy
    study_arousal.optimize(lambda trial: objective(trial, x_train_arousal, x_test_arousal, x_val_arousal,
                                                   y_train_arousal, y_test_arousal, y_val_arousal, 'arousal'),
                           n_trials=150)

    # Plot and save the optimization history
    fig_arousal = vis.plot_optimization_history(study_arousal)
    fig_arousal.update_layout(title="Arousal Optimization History", yaxis_title="MAPE")
    fig_arousal.write_image("./Optuna_History_images/arousal_optuna_history.png")

    # Plot and save the slice plot
    fig_arousal = vis.plot_slice(study_arousal)
    fig_arousal.update_layout(title="Arousal Slice Plot", yaxis_title="MAPE")
    fig_arousal.write_image("./Optuna_History_images/arousal_optuna_slice_plot.png")

    best_trial_arousal = study_arousal.best_trial
    best_learning_rate_arousal = best_trial_arousal.params['learning_rate']
    best_num_units_arousal = best_trial_arousal.params['num_units']
    best_dropout_rate_arousal = best_trial_arousal.params['dropout_rate']
    best_epochs_arousal = best_trial_arousal.params['epochs']
    best_batch_size_arousal = best_trial_arousal.params['batch_size']

    best_metric_arousal, best_model_arousal = train_model(x_train_arousal, x_test_arousal, x_val_arousal,
                                                          y_train_arousal, y_test_arousal, y_val_arousal, 'arousal',
                                                          best_learning_rate_arousal, best_num_units_arousal,
                                                          best_dropout_rate_arousal, best_epochs_arousal,
                                                          best_batch_size_arousal)

    print('Best value: {:.5f}'.format(best_metric_arousal))
    print('Best parameters: {}'.format(best_trial_arousal.params))

    # Save the best mode
    best_model_arousal.save("../Models/arousal_model.h5")


def objective(trial, x_train, x_test, x_val, y_train, y_test, y_val, characteristic):
    # Define the hyperparameters to be optimized
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    num_units = trial.suggest_int('num_units', 32, 512)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    batch_size = trial.suggest_int('batch_size', 8, 128)
    epochs = trial.suggest_int('epochs', 50, 200)

    # Train the model and obtain the validation metric
    metric, _ = train_model(x_train, x_test, x_val, y_train, y_test, y_val, characteristic, learning_rate,
                            num_units, dropout_rate, epochs, batch_size)

    # Return the validation metric as the objective value to be optimized (minimized)
    return metric


def train_model(x_train, x_test, x_val, y_train, y_test, y_val, characteristic, learning_rate, num_units, dropout_rate,
                epochs, batch_size):
    # Define the input shape
    input_shape = (x_train.shape[1],)

    # Define the inputs
    inputs = Input(shape=input_shape)

    # Define the hidden layer with one layer and num_units neurons
    hidden_layer = Dense(num_units, activation='relu')(inputs)
    hidden_layer = Dropout(dropout_rate)(hidden_layer)

    # Define the output layer with 1 neuron
    outputs = Dense(1, activation='sigmoid')(hidden_layer)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model with the desired learning rate
    optimizer = SGD(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=['mean_absolute_percentage_error'])

    # Train the model«
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    print(f"Train MSE: {history.history['loss'][-1]}")
    print(f"Train MAPE: {history.history['mean_absolute_percentage_error'][-1]}")

    # Evaluate the model using the test data
    evaluation_results = model.evaluate(x_test, y_test)
    print(f"Test MSE: {evaluation_results[model.metrics_names.index('loss')]}")
    print(f"Test MAPE: {evaluation_results[model.metrics_names.index('mean_absolute_percentage_error')]}")

    # Validate the model using the validation data
    y_pred = model.predict(x_val)
    mse = metrics.mean_squared_error(y_val, y_pred)
    mape = metrics.mean_absolute_percentage_error(y_val, y_pred)*100
    print(f"Validation MSE: {mse}")
    print(f"Validation MAPE: {mape}")

    # Return the metric values for each epoch during training
    return mape, model


# download_mu_vi_musics()
if __name__ == '__main__':
    if not os.path.exists('Models'):
        os.makedirs('Models')

    build_models()

