import os

import librosa
import numpy as np
import optuna
import pandas as pd
from keras import Model
from keras.layers import Flatten, Dense, BatchNormalization, Dropout
from keras.optimizers import SGD
from numpy import var, mean, max, where
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from keras.layers import Input, Dense, Dropout
import optuna.visualization as vis


def objective(trial, x_train, x_test, x_val, y_train, y_test, y_val):
    # Define the hyperparameters to be optimized
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    num_units = trial.suggest_int('num_units', 250, 1050)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    batch_size = trial.suggest_int('batch_size', 8, 230)
    epochs = trial.suggest_int('epochs', 50, 250)

    # Train the model and obtain the validation metric
    metric, _ = train_model(x_train, x_test, x_val, y_train, y_test, y_val, learning_rate,
                            num_units, dropout_rate, epochs, batch_size)

    # Return the validation metric as the objective value to be optimized (minimized)
    return metric

def build_model():
    music_data = pd.read_csv('file.csv') # 30 seconds musics
    # music_data.head(5)
    # print(music_data)
    # print(music_data['label'].value_counts())

    # Map label from classes to numbers (0 to 9)
    label_encoder = preprocessing.LabelEncoder()
    music_data['label'] = label_encoder.fit_transform(music_data['label'])

    # Define the variables
    X = music_data.drop(['label', 'filename', 'length'], axis=1)
    y = music_data['label']

    # Scale the data
    cols = X.columns
    minmax = preprocessing.MinMaxScaler()
    np_scaled = minmax.fit_transform(X)

    # new data frame with the new scaled data.
    X = pd.DataFrame(np_scaled, columns=cols)

    print("Train and test split...")
    # Split data into training and testing sets for Valence
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Split data into training and validation sets for Valence
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

    # --- Genre Model With Optuna
    print("Training Genre Classifier...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, x_train, x_test, x_val, y_train, y_test, y_val), n_trials=250)

    # Plot and save the optimization history
    fig = vis.plot_optimization_history(study)
    fig.update_layout(title="Genre Optimization History", yaxis_title="Accuracy")
    fig.write_image("./Optuna_History_images/genre_optuna_history.png")

    # Plot and save the slice plot
    fig = vis.plot_slice(study)
    fig.update_layout(title="Genre Slice Plot", yaxis_title="Accuracy")
    fig.write_image("./Optuna_History_images/genre_optuna_slice_plot.png")

    best_trial = study.best_trial
    best_learning_rate = best_trial.params['learning_rate']
    best_num_units = best_trial.params['num_units']
    best_dropout_rate = best_trial.params['dropout_rate']
    best_epochs = best_trial.params['epochs']
    best_batch_size = best_trial.params['batch_size']

    best_metric, best_model = train_model(x_train, x_test, x_val, y_train, y_test, y_val,
                                                          best_learning_rate, best_num_units, best_dropout_rate,
                                                          best_epochs, best_batch_size)

    print('Best validation Accuracy value: {:.5f}'.format(best_metric))
    print('Best parameters: {}'.format(best_trial.params))
    #
    # Save the best model
    best_model.save("../Models/genre_model_2.h5")


def extract_features(directory, csv_filename):
    # Get all files in the directory
    files = os.listdir(directory)

    # music_data = pd.read_csv('file.csv')

    data = []

    for file in files:
        # audio = librosa.load(directory + '/' + file, duration=30.0133)
        # audio = librosa.load(directory + '/' + file, duration=30, offset=60) # 150 seconds -> middle = 75s -> interval = 60 - 90 (30s)

        y, sr = librosa.load(directory + '/' + file, duration=150)

        mfccs = []
        # for each 150 seconds
        for i in range(1, 151):
            mfccs.append(np.mean(librosa.feature.mfcc(y=y[((i-1)*sr):(i*sr)-1], sr=sr)))

        lowest_distance = np.inf
        best_offset = 0

        # mfccs[0:30] -> [1;31] -> [2;32] -> .... [120;150]
        for i in range(0, 121):
            interval = mfccs[i:i+30]
            minimum_distance = np.ptp(interval)

            if minimum_distance <= lowest_distance:
                lowest_distance = minimum_distance
                best_offset = i

        print(best_offset)
        print(lowest_distance)

        audio = librosa.load(directory + '/' + file, duration=30, offset=best_offset) # 150 seconds -> middle = 75s -> interval = 60 - 90 (30s)

        # Length
        # length = audio[0].shape[0]

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

        data.append([file, chroma_stft_mean, chroma_stft_var, rms_mean, rms_var, spectral_centroid_mean,
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
        "filename", "chroma_sftf_mean", "chroma_sftf_var", "rms_mean", "rms_var",
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

def train_model(x_train, x_test, x_val, y_train, y_test, y_val, learning_rate, num_units, dropout_rate,
                epochs, batch_size):

    # model = Sequential()
    #
    # model.add(Flatten(input_shape=(57,)))
    # model.add(Dense(256, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(Dense(10, activation='softmax'))
    #
    # # compile the model
    # adam = keras.optimizers.Adam(lr=1e-4)
    # model.compile(optimizer=adam,
    #               loss="sparse_categorical_crossentropy",
    #               metrics=["accuracy"])
    #
    # hist = model.fit(X_train, y_train,
    #                  validation_data=(X_test, y_test),
    #                  epochs=100,
    #                  batch_size=32)
    #
    # model_to_save = hist.model
    #
    # # Save the best model
    # model_to_save.save("../Models/genre_model.h5")
    #
    # test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    # print(f"Test accuracy: {test_accuracy}")
    # --------------------------------

    # Define the input shape
    input_shape = (x_train.shape[1],)

    # Define the inputs
    inputs = Input(shape=input_shape)

    # Define the hidden layer with one layer and num_units neurons
    hidden_layer = Dense(num_units, activation='sigmoid')(inputs)
    hidden_layer = Dropout(dropout_rate)(hidden_layer)

    # Define the output layer with 1 neuron
    outputs = Dense(10, activation='softmax')(hidden_layer)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model with the desired learning rate
    optimizer = SGD(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    print(f"Train Sparse Categorical crossentropy: {history.history['loss'][-1]}")
    print(f"Train Accuracy: {history.history['accuracy'][-1]}")

    # Evaluate the model using the test data
    evaluation_results = model.evaluate(x_test, y_test)
    print(f"Test Sparse Categorical crossentropy: {evaluation_results[model.metrics_names.index('loss')]}")
    print(f"Test Accuracy: {evaluation_results[model.metrics_names.index('accuracy')]}")

    # Validate the model using the validation data
    y_pred = []
    estimated_pred = model.predict(x_val)

    for i in range(0, len(estimated_pred)):
        y_pred.append(np.argmax(estimated_pred[i]))

    accuracy_score = metrics.accuracy_score(y_val, y_pred, normalize=True)
    f1_score = metrics.f1_score(y_val, y_pred, average='micro')
    print(f"Validation Accuracy Score: {accuracy_score}")
    print(f"Validation F1 Score: {f1_score}")

    # Return the metric values for each epoch during training
    return accuracy_score, model

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
    genre_model = keras.models.load_model(f'../Models/genre_model_4.h5')
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
    build_model()
    # extract_features('../musics', '../musics_genre_features_6.csv')
    # predict_musics_genres('../musics', '../GenreClassification/FeaturesExtracted/musics_genre_features.csv',
    #                       '../GenreClassification/GenresPredicted/musics_genre_predicted.csv')
    # predict_musics_genres('../musics', '../musics_genre_features_2.csv', '../musics_genre_predicted_2.csv')