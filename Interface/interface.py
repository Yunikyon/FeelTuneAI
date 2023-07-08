import csv
import math
import os
import shutil
import warnings
from datetime import datetime
from time import sleep

import cv2
import joblib
import keras.optimizers
import librosa
import numpy as np
import pandas as pd
import tf as tf
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from mutagen.mp3 import MP3
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import SGD
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import model_from_json
import optuna
import optuna.visualization as vis

import context.main as contextMain
from EmotionRecognition.EmotionDetection import capture_emotion

import pygame as pygame
from PyQt5.QtCore import QSize, Qt, QPoint, QTimer, QRect, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QPalette, QColor, QIcon, QCursor, QPainter, QPen, QFontMetrics, QKeyEvent, QMovie
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMainWindow, QLabel, QLineEdit, QVBoxLayout, \
    QHBoxLayout, QSlider, QMessageBox, QStackedWidget, QFileDialog
from numpy.core.defchararray import strip
import random

from download_from_yt import download_musics, download_musics_from_csv
from predict_musics_VA import predict_music_directory_emotions, predict_uploaded_music_emotions

current_user_name = ''
is_in_building_dataset_phase = True
current_user_bpd_progress = 0
music_files_bdp_length = 0
has_user_finished_first_iteration_of_bdp = False
musics_listened_by_current_user = [] # To choose what music to play next
musics_listened_by_current_user_in_current_session = []
last_context_data = ""
last_time_context_data_was_called = ""

# dataset for model training variables
data = []
current_music_emotions = ''
new_record = {'date': '', 'initial_emotion': '', 'music_name': '', 'last_emotion': '', 'rated_emotion': '',
              'instant_seconds|percentages|dominant_emotion': ''}

goal_emotion = None
rated_emotion = None
valence_arousal_pairs = None
application_music_names = None

is_training_model = False

def reset_values(record):
    record['initial_emotion'] = ''
    record['music_name'] = ''
    record['average_emotion'] = ''

    return record


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


class Color(QWidget):
    def __init__(self, color):
        super(Color, self).__init__()
        self.setAutoFillBackground(True)

        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(color))
        self.setPalette(palette)


class CircleProgressWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.progress = 0
        self.duration = 0

        # Configure Pygame Mixer
        pygame.mixer.init()

        # Create a timer to update the progress every second
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(1000)

    def set_duration(self, duration):
        self.duration = duration

    def set_progress(self, progress):
        self.progress = progress
        self.update()

    def update_progress(self):
        # Get the current position of the music playback
        position = pygame.mixer.music.get_pos() // 1000
        self.set_progress(position)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Set the outer circle parameters
        # outer_radius = min(self.width(), self.height()) / 2 - 10
        outer_radius = min(250, 250) / 2 - 10
        outer_center = self.rect().center()

        # Set the inner circle parameters
        inner_radius = outer_radius - 20
        inner_center = outer_center

        # Draw the outer circle
        pen = QPen(QColor(207, 186, 163))
        pen.setWidth(10)
        painter.setPen(pen)
        painter.drawEllipse(outer_center, outer_radius, outer_radius)

        # Draw the progress arc
        pen.setColor(QColor(247, 201, 151))
        painter.setPen(pen)
        start_angle = 90 * 16  # 90 degrees in 1/16th of a degree
        span_angle = -self.progress * 360 * 16 / self.duration
        painter.drawArc(int(inner_center.x() - inner_radius), int(inner_center.y() - inner_radius),
                        int(inner_radius * 2), int(inner_radius * 2), int(start_angle), int(span_angle))

    def resizeEvent(self, event):
        self.update()


def confirm_warning(self, title, message):
    reply = QMessageBox.warning(
        self, title, message,
        QMessageBox.Yes | QMessageBox.No,
    )
    return reply

def convert_emotions_to_va_values(emotion):
    # Example -> 6|45.581-0.0-0.0-1.344-47.149-0.0-5.925|sad
    percentages = emotion.split('|')[1]

    # 1. Get percentages
    angry, disgust, fear, happy, sad, surprise, neutral = percentages.split('-')
    angry, disgust, fear, happy, sad, surprise, neutral = float(angry), float(disgust), float(fear), float(happy), float(sad), float(surprise), float(neutral)

    # 2. Define the limits of the graphic percentage space
    high = 0.84
    medium = 0.5
    low = 0.16

    # 3. Calculate and get valence and arousal
    valence_sum = surprise * medium + neutral * medium + sad * low + happy * high + fear * low + disgust * low + angry * low
    arousal_sum = surprise * high + neutral * medium + sad * medium + happy * medium + fear * high + disgust * medium + angry * high

    # 4. Convert from the range [0; 100] to the range [0; 1]
    valence_sum = convert_to_new_range(0, 100, -1, 1, valence_sum)
    arousal_sum = convert_to_new_range(0, 100, -1, 1, arousal_sum)

    return valence_sum, arousal_sum


def convert_to_new_range(input_min, input_max, new_min, new_max, value_to_convert):
    # --- Convert to range [-1, 1] from [0, 100] ---
    conversion_factor = (new_max - new_min) / (input_max - input_min)
    offset = new_min - input_min * conversion_factor

    value_converted = value_to_convert * conversion_factor + offset

    return value_converted


def get_context():
    dataframe = contextMain.execute()
    if dataframe is not None:
        csv_context = dataframe.to_csv(index=False)
        context_headers = csv_context.split('\n')[0]
        context_headers = context_headers.split(',')

        number_of_headers = len(context_headers)
        context_values = csv_context.split('\n')[1]
        context_values = context_values.split(',')

        context_dict = {}
        for header, value in zip(context_headers, context_values):
            header = header.rstrip('\r')
            value = str(value).rstrip('\r')
            context_dict.update({header: value})
        # context_dict = {header: str(value).rstrip('\r') for header, value in zip(context_headers, context_values)}
        return context_dict, number_of_headers
    return {}, 0


def merge_musics_va_to_dataset(dataset):
    delimiter = '~~~'
    with open('../building_dataset_phase_musics_va.csv', 'r') as file_obj:
        try:
            musics_df = pd.read_csv(file_obj, sep=delimiter, engine='python')
        except pd.errors.ParserError:
            musics_df = pd.read_csv(file_obj.replace(delimiter, ','), sep=',')
    dataset = pd.merge(dataset, musics_df, on='music_name', how='left')

    for index,row in dataset.iterrows():
        dataset.at[index, 'music_valence'] = convert_to_new_range(-1, 1, 0, 1, dataset.at[index, 'music_valence'])
        dataset.at[index, 'music_arousal'] = convert_to_new_range(-1, 1, 0, 1, dataset.at[index, 'music_arousal'])

    return dataset


def add_va_columns_from_emotions(dataset):
    global is_training_model

    for index, row in dataset.iterrows():
        # Example -> 6|45.581-0.0-0.0-1.344-47.149-0.0-5.925|sad;9|45.581-0.0-0.0-1.344-47.149-0.0-5.925|sad;
        emotions = dataset.at[index, 'instant_seconds|percentages|dominant_emotion']

        first_emotion = emotions.split(';')[0]

        if is_training_model:  # Last emotion
            # 1. Get last emotion (percentages)
            last_emotion = emotions.split(';')[-2]

            # 2. Convert last emotion's percentages to valence and arousal
            dataset.at[index, 'valence_last_emotion'],\
                dataset.at[index, 'arousal_last_emotion'] = convert_emotions_to_va_values(last_emotion)

            # 3. Apply rated emotion weight
            rated_emotion = dataset.at[index, 'rated_emotion'].split('|')
            rated_emotion_0 = float(rated_emotion[0])
            rated_emotion_1 = float(rated_emotion[1])

            if len(rated_emotion) != 2:
                continue

            # 3. Apply rated emotion weight
            dataset.at[index, 'valence_last_emotion'] = (dataset.at[index, 'valence_last_emotion'] + rated_emotion_0) / 2
            dataset.at[index, 'arousal_last_emotion'] = (dataset.at[index, 'arousal_last_emotion'] + rated_emotion_1) / 2
        else:  # Goal emotion
            global goal_emotion
            dataset.at[index, 'valence_last_emotion'] = goal_emotion[0]
            dataset.at[index, 'arousal_last_emotion'] = goal_emotion[1]

        # Convert first emotion's percentages to valence and arousal
        dataset.at[index, 'valence_initial_emotion'],\
            dataset.at[index, 'arousal_initial_emotion'] = convert_emotions_to_va_values(first_emotion)

    return dataset

def one_hot_encoding(filtered_df, filtered_df_column_name, predefined_columns):
    # 1. One hot encode existing values
    one_hot_encoded = pd.get_dummies(filtered_df[filtered_df_column_name], columns=predefined_columns, prefix_sep=' = ',
                                     prefix=filtered_df_column_name)

    # 2. Add missing columns with all values equal to 0
    missing_cols = set([f'{filtered_df_column_name} = {column}' for column in predefined_columns]) - set(one_hot_encoded.columns)
    for col in missing_cols:
        one_hot_encoded[col] = 0

    return one_hot_encoded

def normalize_dataset(filtered_df):
    global is_training_model

    if is_training_model:  # if training, not predicting
        filtered_df = filtered_df.drop(labels=['username', 'last_emotion'], axis=1)
        filtered_df = merge_musics_va_to_dataset(filtered_df)
        filtered_df = filtered_df.drop(labels=['music_name'], axis=1)

    if filtered_df is None:
        # TODO - mostrar erro
        return

    mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

    def convert_hour_to_minutes(time):
        if time is None or time == '':
            return

        time = str(time)
        if time == 'nan':
            return

        hours, minutes, seconds = map(int, time.split(':'))
        hours_in_minutes = hours * 60
        total_minutes = hours_in_minutes + minutes
        return total_minutes

    def min_max_normalization(value, min, max):
        return (value - min) / (max - min)

    # --------------- Create a new normalized dataframe for model training ---------------

    # --- Time columns: listenedAt, sunset, sunrise, day_length ---
    hours_columns = ['listenedAt', 'sunset', 'sunrise', 'day_length']

    # TODO - remove ?
    # hours_scaler = MinMaxScaler(feature_range=(0, 1))
    # filtered_df[hours_columns] = hours_scaler.fit_transform(filtered_df[hours_columns])

    # For every time column normalize with min max
    for column in hours_columns:
        # 1. Convert time to minutes
        filtered_df[column] = filtered_df[column].apply(convert_hour_to_minutes)

        # 2. Replace missing values with average
        filtered_df[column] = mean_imputer.fit_transform(filtered_df[column].array.reshape(-1, 1))

        # 3. Convert the range [0; 1440] to the range [0; 1]
        min_value = 0
        max_value = 1440  # 1440/60 = 24 hours
        filtered_df[column] = min_max_normalization(filtered_df[column], min_value, max_value)

    # --- Column: isWorkDay - Map to binary (no = 0 : yes = 1) ---
    filtered_df['isWorkDay'] = filtered_df['isWorkDay'].map({"Yes": 1, "No": 0})

    # Replace missing isWorkDay values with most frequent
    array_1d_isWorkDay = np.array(filtered_df['isWorkDay'])
    array_2d_isWorkDay = array_1d_isWorkDay.reshape(-1, 1)
    filtered_df['isWorkDay'] = mode_imputer.fit_transform(array_2d_isWorkDay).flatten()

    # --- Categorical columns: initial_emotion, idWeatherType, classWindSpeed, classPrecInt, timeOfDay ---
    categorical_columns = ['initial_emotion', 'idWeatherType',
                           'classWindSpeed', 'classPrecInt', 'timeOfDay']

    # Replace missing values with most frequent
    mode_imputer.fit(filtered_df[categorical_columns])
    filtered_df[categorical_columns] = mode_imputer.transform(filtered_df[categorical_columns])

    # --- One Hot Encoding for categorical variables
    # idWeatherType one hot encoding
    filtered_df = pd.concat([filtered_df, one_hot_encoding(filtered_df, 'idWeatherType', ['No information', 'Clear sky',
                                                                                          'Partly cloudy', 'Sunny intervals',
                                                                                          'Cloudy ', 'Cloudy (High cloud)',
                                                                                          'Showers/rain', 'Light showers/rain',
                                                                                          'Heavy showers/rain', 'Rain/showers',
                                                                                          'Light rain', 'Heavy rain/showers',
                                                                                          'Intermittent rain',
                                                                                          'Intermittent ligth rain',
                                                                                          'Intermittent heavy rain', 'Drizzle',
                                                                                          'Mist', 'Fog', 'Snow', 'Thunderstorms',
                                                                                          'Showers and thunderstorms', 'Hail',
                                                                                          'Frost', 'Rain and thunderstorms',
                                                                                          'Convective clouds', 'Partly cloudy',
                                                                                          'Fog', 'Cloudy', 'Snow showers',
                                                                                          'Rain and snow'])], axis=1)

    # classWindSpeed one hot encoding
    filtered_df = pd.concat([filtered_df, one_hot_encoding(filtered_df, 'classWindSpeed', ['Weak', 'Moderate', 'Strong',
                                                                                           'Very', 'Strong'])], axis=1)

    # classPrecInt one hot encoding
    filtered_df = pd.concat([filtered_df, one_hot_encoding(filtered_df, 'classPrecInt', ['No precipitation', 'Weak',
                                                                                         'Moderate', 'Strong'])], axis=1)

    # timeOfDay one hot encoding
    filtered_df = pd.concat([filtered_df, one_hot_encoding(filtered_df, 'timeOfDay', ['Night', 'Early Morning', 'Morning',
                                                                                      'Afternoon', 'Evening'])], axis=1)

    # initial_emotion one hot encoding
    filtered_df = pd.concat(
        [filtered_df, one_hot_encoding(filtered_df, 'initial_emotion', ['angry', 'fear', 'disgust', 'sad', 'neutral',
                                                                        'surprise', 'happy'])], axis=1)

    # drop labels used for one hot encoding, they will not be used anymore
    filtered_df = filtered_df.drop(labels=['idWeatherType', 'classWindSpeed', 'classPrecInt', 'timeOfDay',
                                           'initial_emotion'], axis=1)

    # --- Numerical columns: tMin, tMax, temp, feels_like, cloud_pct, humidity, wind_speed,
    # precipitaProb ---
    numerical_columns = ['tMin', 'tMax', 'temp',
                         'feels_like', 'cloud_pct', 'humidity',
                         'wind_speed', 'precipitaProb']

    # Replacing missing values with average
    mean_imputer.fit(filtered_df[numerical_columns])  # finds the mean of every column
    filtered_df[numerical_columns] = mean_imputer.transform(filtered_df[numerical_columns])  # replaces

    # --- Temperature columns: tMin, tMax, temp, feels_like ---
    temperature_columns = ['tMin', 'tMax', 'temp', 'feels_like']

    # For every temperature column normalize with min max
    for column in temperature_columns:
        # Convert the range [-20; 50] to the range [0; 1]
        min_value = -20
        max_value = 50
        filtered_df[column] = min_max_normalization(filtered_df[column], min_value, max_value)

    # TODO - remover ?
    # temp_scaler = MinMaxScaler(feature_range=(0, 1))
    # filtered_df[temperature_columns] = temp_scaler.fit_transform(filtered_df[temperature_columns])

    # --- Percentage columns: cloud_pct, precipitaProb, humidity
    percentage_columns = ['cloud_pct', 'precipitaProb', 'humidity']

    # For every percentage column normalize with min max
    for column in percentage_columns:
        # Convert the range [0; 100] to the range [0; 1]
        min_value = 0
        max_value = 100
        filtered_df[column] = min_max_normalization(filtered_df[column], min_value, max_value)

    # TODO - remover ?
    # percentage_scaler = MinMaxScaler(feature_range=(0, 1))
    # filtered_df[from_0_to_100_values] = percentage_scaler.fit_transform(filtered_df[from_0_to_100_values])

    # TODO - remover ?
    # --- Column: wind_speed ---
    # wind_speed_scaler = MinMaxScaler(feature_range=(0, 1))
    # filtered_df['wind_speed'] = wind_speed_scaler.fit_transform(filtered_df[['wind_speed']])

    # --- Column: wind_speed ---
    for index, row in filtered_df.iterrows():
        # Convert the range [0; 75] to the range [0; 1]
        min_value = 0
        max_value = 75
        filtered_df.at[index, 'wind_speed'] = min_max_normalization(row['wind_speed'], min_value, max_value)

    # TODO - remover ?
    # Apply scaler
    # scaler = StandardScaler()
    # filtered_df[numerical_columns] = scaler.fit_transform(filtered_df[numerical_columns])

    # Convert first and last emotions from percentages to valence and arousal
    filtered_df = add_va_columns_from_emotions(filtered_df)

    # VA columns: valence_last_emotion, arousal_last_emotion, valence_initial_emotion, arousal_initial_emotion
    va_columns = ['valence_last_emotion', 'arousal_last_emotion', 'valence_initial_emotion', 'arousal_initial_emotion']

    for column in va_columns:
        # Convert the range [-1; 1] to the range [0; 1]
        min_value = -1
        max_value = 1
        filtered_df[column] = min_max_normalization(filtered_df[column], min_value, max_value)

    filtered_df = filtered_df.drop(labels=['instant_seconds|percentages|dominant_emotion'], axis=1)

    if is_in_building_dataset_phase:
        filtered_df = filtered_df.drop(labels=['rated_emotion'], axis=1)

    numerical_columns.extend(['listenedAt', 'sunrise', 'sunset', 'day_length', 'valence_initial_emotion',
                              'arousal_initial_emotion', 'valence_last_emotion', 'arousal_last_emotion'])

    if is_training_model: # Add labels to train
        numerical_columns.extend(['music_valence', 'music_arousal'])

    # TODO - remover ?
    # Discretize the numerical columns into categories
    # num_bins = 7  # Number of bins or categories
    # for col in numerical_columns:
    #     filtered_df[col] = pd.cut(filtered_df[col], bins=num_bins, labels=False)

    filtered_df[numerical_columns] = filtered_df[numerical_columns].round(3)

    return filtered_df



class LoginWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("FeelTuneAI")
        self.setMouseTracking(True)
        self.setMinimumSize(QSize(1200, 750))

        # Base Layout
        base_layout = QHBoxLayout()
        base_layout.setContentsMargins(0, 0, 0, 0)
        base_layout.setSpacing(0)

        # Left Layout
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        feeltune_label = QLabel("FeelTune\n\tAI")
        font = feeltune_label.font()
        font.setPointSize(35)
        feeltune_label.setFont(font)
        feeltune_label.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        feeltune_label.setContentsMargins(0, 80, 0, 30)
        left_layout.addWidget(feeltune_label)

        logo = QLabel()
        logo.setPixmap(QPixmap('./images/feeltuneAI_logo.png'))
        logo.setScaledContents(True)
        logo.setMaximumSize(350, 420)
        logo.setContentsMargins(0, 0, 20, 120)
        logo.setAlignment(Qt.AlignHCenter)
        left_layout.addWidget(logo)

        left_widget = Color('white')
        left_widget.setLayout(left_layout)
        left_widget.setMinimumSize(QSize(420, 750))
        left_widget.setMaximumSize(QSize(420, 2000))

        # Right Layout
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)

        tune_in_label = QLabel("Tune In!")
        font = tune_in_label.font()
        font.setPointSize(35)
        tune_in_label.setFont(font)
        tune_in_label.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        tune_in_label.setContentsMargins(0, 90, 0, 0)
        tune_in_label.setMaximumSize(2000, 230)
        right_layout.addWidget(tune_in_label)

        # Input Name layout
        input_name_layout = QHBoxLayout()
        input_name_layout.setAlignment(Qt.AlignHCenter)
        input_name_layout.setContentsMargins(0, 0, 40, 0)

        name_label = QLabel("Name: ")
        font_name = name_label.font()
        font_name.setPointSize(20)
        name_label.setFont(font_name)
        name_label.setAlignment(Qt.AlignRight)
        name_label.setMaximumSize(120, 200)
        input_name_layout.addWidget(name_label)

        self.input_name = QLineEdit()
        input_name_font = self.input_name.font()
        input_name_font.setPointSize(15)
        self.input_name.setFont(input_name_font)
        self.input_name.setMaxLength(50)
        self.input_name.setPlaceholderText("\tEnter your name")
        self.input_name.setMaximumSize(500, 50)
        input_name_layout.addWidget(self.input_name)

        input_name_widget = QWidget()
        input_name_widget.setLayout(input_name_layout)
        input_name_widget.setMaximumSize(2000, 50)
        right_layout.addWidget(input_name_widget)

        # Blank space one
        blank_space_one = QLabel()
        blank_space_one.setMaximumSize(10, 30)
        right_layout.addWidget(blank_space_one)

        # Button
        button_layout = QHBoxLayout()
        button_layout.setAlignment(Qt.AlignHCenter)

        self.tune_in_button = QPushButton(" Enter")
        tune_in_font = self.tune_in_button.font()
        tune_in_font.setPointSize(20)
        self.tune_in_button.setFont(tune_in_font)
        self.tune_in_button.setMinimumSize(60, 60)
        self.tune_in_button.setIcon(QIcon('./images/tune_in_btn.png'))
        self.tune_in_button.setIconSize(QSize(50, 50))
        self.tune_in_button.setFlat(True)
        self.tune_in_button.setStyleSheet("QPushButton { background-color: transparent;}")
        self.tune_in_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.tune_in_button.clicked.connect(self.show_next_window)
        button_layout.addWidget(self.tune_in_button)

        button_widget = QWidget()
        button_widget.setLayout(button_layout)
        button_widget.setMaximumSize(2000, 90)
        right_layout.addWidget(button_widget)

        # Blank space two
        blank_space_two = QLabel()
        blank_space_two.setMaximumSize(10, 1200)
        right_layout.addWidget(blank_space_two)

        right_widget = Color('#f5e6d0')
        right_widget.setLayout(right_layout)

        # Add layouts to base
        base_layout.addWidget(left_widget)
        base_layout.addWidget(right_widget)

        base_widget = QWidget()
        base_widget.setLayout(base_layout)
        self.setCentralWidget(base_widget)

    def show_next_window(self, ):
        global current_user_name
        current_user_name = self.input_name.text()

        if current_user_name == "":
            QMessageBox.warning(
                self, "Error", "Name can't be empty",
                QMessageBox.Ok,
            )
            return
        if ',' in current_user_name:
            QMessageBox.warning(
                self, "Error", "Name can't contain a comma character",
                QMessageBox.Ok,
            )
            return

        global musics_listened_by_current_user
        global is_in_building_dataset_phase
        global current_user_bpd_progress

        progress = 0
        aux_musics_listened_previously = ''
        file_exists = os.path.isfile('../users.csv')

        # Read user's save information
        if file_exists:
            with open('../users.csv', 'r', encoding='utf-8') as file:
                for i, line in enumerate(file):
                    if i == 0:
                        continue
                    line_content = line.split('~~~')
                    user_name = line_content[0]
                    if user_name == current_user_name.lower():
                        progress = int(line_content[1])
                        aux_musics_listened_previously = line_content[2]
                        break

        if progress == 100:
            is_in_building_dataset_phase = False

        if aux_musics_listened_previously != '':
            musics_listened_by_current_user = aux_musics_listened_previously.replace('\n', '').split('___')

        current_user_bpd_progress = progress

        if is_in_building_dataset_phase:
            self.nextWindow = BuildingPhaseHomeScreen()
            self.nextWindow.show()
            self.close()
        else:
            self.nextWindow = ApplicationHomeScreen()
            self.nextWindow.show()
            self.close()


class MusicsWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("FeelTuneAI")
        self.setMouseTracking(True)
        self.setMinimumSize(QSize(1200, 750))

        self.nextWindow = None
        self.initial_emotion = None

        global is_in_building_dataset_phase
        global current_user_bpd_progress
        global current_user_name

        self.music_playing = False
        self.is_rating_music = False

        self.stacked_widget = QStackedWidget()

        self.music_is_paused = False

        # Music Thread Initialization
        if is_in_building_dataset_phase:
            musics_directory = '../BuildingDatasetPhaseMusics'
        else:
            musics_directory = '../ApplicationMusics'

        # --- Get music files and choose music
        self.music_files = []
        self.music_files_length = 0

        self.music_files = os.listdir(musics_directory)
        personalized_directory = ""
        # Read training musics
        if is_in_building_dataset_phase:
            # Add personalized bdp user musics to base musics
            if os.path.exists(f'../personalized_musics/{current_user_name}/building_dataset_phase_musics'):
                personalized_directory = f'../personalized_musics/{current_user_name}/building_dataset_phase_musics'
                personal_music_files = os.listdir(
                    f'../personalized_musics/{current_user_name}/building_dataset_phase_musics')
                self.music_files.extend(personal_music_files)

            self.music_files_length = len(self.music_files)

            # Verify if doesn't exit musics
            global music_files_bdp_length
            music_files_bdp_length = self.music_files_length

            if self.music_files_length == 0:
                print(f"{Bcolors.WARNING} BDP music files length is zero" + Bcolors.ENDC)
                exit()
        else:  # Read application musics

            # TODO - é recolher contexto, emoção atual da pessoa,
            #  emoção desejada da pessoa para escolher a música, baseado no treino da rede neuronal :)
            # Add personalized application user musics to base musics
            if os.path.exists(f'../personalized_musics/{current_user_name}/application_musics'):
                personalized_directory = f'../personalized_musics/{current_user_name}/application_musics'
                personal_music_files = os.listdir(f'../personalized_musics/{current_user_name}/application_musics')
                self.music_files.extend(personal_music_files)

            # Verify if doesn't exit musics
            self.music_files_length = len(self.music_files)
            if self.music_files_length == 0:
                print(f"{Bcolors.WARNING} Application music files length is zero" + Bcolors.ENDC)
                exit()

        self.music_thread = MusicThread(musics_directory, personalized_directory)
        self.music_thread.set_music("NA")
        self.music_thread.finished_music_signal.connect(self.music_finished)

        # Emotion Thread Initialization
        self.emotion_thread = EmotionsThread()
        self.emotion_thread.captured_one_emotion.connect(self.emotion_captured)

        # Base Layout
        base_layout = QVBoxLayout()
        base_layout.setContentsMargins(10, 20, 10, 10)
        base_layout.setSpacing(0)

        header_line = QHBoxLayout()
        header_line.setContentsMargins(0, 0, 0, 0)

        # Title Layout
        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)

        logo = QLabel()
        logo.setPixmap(QPixmap('./images/feeltuneAI_logo.png'))
        logo.setScaledContents(True)
        logo.setMaximumSize(70, 70)
        title_layout.addWidget(logo)

        feeltune_ai_label = QLabel('FeelTune AI')
        feeltune_ai_font = feeltune_ai_label.font()
        feeltune_ai_font.setPixelSize(30)
        feeltune_ai_label.setFont(feeltune_ai_font)
        feeltune_ai_label.setMinimumSize(960, 80)
        title_layout.addWidget(feeltune_ai_label)

        title_widget = QWidget()
        title_widget.setLayout(title_layout)
        header_line.addWidget(title_widget)

        if is_in_building_dataset_phase:
            # Training Phase corner
            corner_layout = QHBoxLayout()
            corner_layout.setAlignment(Qt.AlignRight)

            line_layout = QHBoxLayout()
            lines = self.PaintLine()
            lines.setLayout(line_layout)
            lines.setMinimumSize(40, 80)
            corner_layout.addWidget(lines)

            # BDP Label
            training_label = QLabel('Building Dataset\n \t Phase')
            training_font = training_label.font()
            training_font.setPixelSize(25)
            training_label.setFont(training_font)
            corner_layout.addWidget(training_label)

            corner_widget = QWidget()
            corner_widget.setLayout(corner_layout)
            header_line.addWidget(corner_widget)

        header_line_widget = QWidget()
        header_line_widget.setLayout(header_line)
        header_line_widget.setMaximumSize(2000, 80)
        base_layout.addWidget(header_line_widget)

        # Blank space one
        blank_space_one = QLabel()
        blank_space_one.setMaximumSize(10, 30)
        base_layout.addWidget(blank_space_one)

        # --- Animation widget
        animation_layout = QVBoxLayout()
        animation_layout.setAlignment(Qt.AlignHCenter)

        if is_in_building_dataset_phase:
            # Training Progress Slider
            progress_layout_vertical = QVBoxLayout()
            progress_layout_vertical.setAlignment(Qt.AlignHCenter)

            # Slider value
            slider_line_layout = QHBoxLayout()
            slider_line_layout.setContentsMargins(0, 0, 0, 0)
            slider_line_layout.setSpacing(20)

            slider_blank_space = QLabel()
            slider_blank_space_font = slider_blank_space.font()
            slider_blank_space_font.setPointSize(15)
            slider_blank_space.setFont(slider_blank_space_font)
            slider_blank_space.setMaximumSize(100, 30)
            slider_line_layout.addWidget(slider_blank_space)

            self.slider_value_label_animation = QLineEdit(str(current_user_bpd_progress) + "%")
            self.slider_value_label_animation.setReadOnly(True)

            slider_font = self.slider_value_label_animation.font()
            slider_font.setPointSize(13)
            self.slider_value_label_animation.setFont(slider_font)

            self.slider_value_label_animation.setStyleSheet("* { background-color: rgba(0, 0, 0, 0); border: rgba(0, 0, 0, 0); z-index: 1}");
            self.slider_value_label_animation.setMaximumSize(810, 50)
            self.slider_value_label_animation.setMinimumSize(70, 50)

            self.slider_value_label_animation.textChanged.connect(self.move_slider_label)
            slider_line_layout.addWidget(self.slider_value_label_animation)

            slider_line_widget = QWidget()
            slider_line_widget.setLayout(slider_line_layout)
            slider_line_widget.setMaximumSize(1000, 30)
            slider_line_widget.setMinimumSize(1000, 30)
            progress_layout_vertical.addWidget(slider_line_widget)

            progress_line_layout = QHBoxLayout()
            progress_line_layout.setContentsMargins(0, 0, 0, 0)
            progress_line_layout.setSpacing(20)

            progress_label = QLabel("Progress")
            progress_font = progress_label.font()
            progress_font.setPointSize(15)
            progress_label.setFont(progress_font)
            progress_label.setMaximumSize(100, 60)
            progress_label.setMinimumSize(100, 60)
            progress_line_layout.addWidget(progress_label)

            self.progress_slider_animation = QSlider(Qt.Horizontal)
            self.progress_slider_animation.setMinimum(0)
            self.progress_slider_animation.setValue(current_user_bpd_progress)
            self.progress_slider_animation.setMaximum(100)
            # self.progress_slider.setSingleStep(round(100/self.music_files_length))
            self.progress_slider_animation.setMaximumSize(800, 40)
            self.progress_slider_animation.setStyleSheet("QSlider::groove:horizontal "
                                          "{border: 1px solid #999999; height: 8px;"
                                            "margin: 2px 0;} "
                                          "QSlider::handle:horizontal "
                                          "{background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #f7c997, stop:1 #ffffff);"
                                            "border: 1px solid #f7c997; width: 0px;"
                                            "margin: -5px 0; border-radius: 3px;}"
                                          "QSlider::add-page:horizontal {background: white}"
                                          "QSlider::sub-page:horizontal {background: #ffd7ab}")
            self.progress_slider_animation.valueChanged.connect(self.slider_value_changed)
            self.progress_slider_animation.setEnabled(False)
            progress_line_layout.addWidget(self.progress_slider_animation)

            self.slider_value_label_animation.setMinimumSize(self.progress_slider_animation.width()+100, 30)

            self.slider_value_label_animation.setContentsMargins(int(current_user_bpd_progress * 7.7), 10, 0, 0)

            progress_line_widget = QWidget()
            progress_line_widget.setLayout(progress_line_layout)
            progress_line_widget.setMaximumSize(1000, 80)
            progress_line_widget.setMinimumSize(1000, 80)

            progress_layout_vertical.addWidget(progress_line_widget)
            progress_layout_vertical_widget = QWidget()
            progress_layout_vertical_widget.setMaximumSize(2000, 110)
            progress_layout_vertical_widget.setLayout(progress_layout_vertical)

            animation_layout.addWidget(progress_layout_vertical_widget)

        # Circles animation
        circle_layout = QHBoxLayout()
        circle_layout.setAlignment(Qt.AlignHCenter)

        # Create a label to display the GIF
        music_progress_layout = QHBoxLayout()
        music_progress_layout.setAlignment(Qt.AlignHCenter)

        blank_space = QLabel()
        blank_space.setMinimumSize(150, 10)
        blank_space.setMaximumSize(150, 10)
        circle_layout.addWidget(blank_space)

        self.music_progress = CircleProgressWidget()
        self.music_progress.setMaximumSize(400, 400)
        self.music_progress.setMinimumSize(400, 400)
        music_progress_layout.addWidget(self.music_progress)

        music_progress_widget = QWidget()
        music_progress_widget.setLayout(music_progress_layout)
        music_progress_widget.setMaximumSize(600, 400)
        music_progress_widget.setMinimumSize(600, 400)
        circle_layout.addWidget(music_progress_widget)

        volume_slider_layout = QHBoxLayout()
        volume_slider_layout.setAlignment(Qt.AlignRight)

        volume_column_layout = QVBoxLayout()

        # Volume slider
        self.volume_slider = QSlider(Qt.Vertical)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setValue(20)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setSingleStep(5)
        self.volume_slider.setStyleSheet("QSlider::groove:vertical "
                                           "{border: 1px solid #999999; width: 5px;"
                                           "margin: 2px 0;} "
                                           "QSlider::handle:vertical "
                                           "{background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #6a6a6a, stop:1 #4d4d5d);"
                                           "border: 1px solid #4d4d5d;  height: 8px;"
                                           "margin: 0px -8px; border-radius: 3px;}"
                                           "QSlider::sub-page:vertical {background: #a2a2a2}"
                                           "QSlider::add-page:vertical {background: #4d4d5d}")
        self.volume_slider.valueChanged.connect(self.volume_slider_value_changed)
        self.volume_slider.setMaximumSize(30, 250)
        self.volume_slider.setMinimumSize(30, 250)
        volume_column_layout.addWidget(self.volume_slider)

        # Volume Icon
        volume_icon = QLabel()
        volume_icon.setPixmap(QPixmap('./images/Speaker_Icon.svg.png'))
        volume_icon.setScaledContents(True)
        volume_icon.setMaximumSize(30, 30)
        volume_column_layout.addWidget(volume_icon)

        volume_column_widget = QWidget()
        volume_column_widget.setLayout(volume_column_layout)
        volume_slider_layout.addWidget(volume_column_widget)

        volume_slider_widget = QWidget()
        volume_slider_widget.setLayout(volume_slider_layout)
        volume_slider_widget.setMaximumSize(200, 400)
        volume_slider_widget.setMinimumSize(200, 400)
        circle_layout.addWidget(volume_slider_widget)

        circle_widget = QWidget()
        circle_widget.setLayout(circle_layout)
        circle_widget.setMaximumSize(2000, 400)

        animation_layout.addWidget(circle_widget)

        # Buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(0)

        buttons_left_layout = QHBoxLayout()
        buttons_left_layout.setContentsMargins(0, 0, 0, 0)
        buttons_left_layout.setSpacing(15)
        buttons_left_layout.setAlignment(Qt.AlignLeft)

        # Button Quit
        quit_button = QPushButton("Quit")
        quit_button_font = quit_button.font()
        quit_button_font.setPointSize(10)
        quit_button.setFont(quit_button_font)
        quit_button.setMaximumSize(120, 60)
        quit_button.setMinimumSize(120, 60)
        quit_button.setIcon(QIcon("./images/quit_btn.png"))
        quit_button.setIconSize(QSize(35, 35))
        quit_button.setCursor(QCursor(Qt.PointingHandCursor))
        quit_button.setStyleSheet("* {background-color: #cfbaa3; border: 1px solid black;} *:hover {background-color: #ba9a75;}")
        quit_button.clicked.connect(self.quit_button_clicked)
        buttons_left_layout.addWidget(quit_button)

        # Button Sign Out
        sign_out_button = QPushButton(" Sign Out")
        sign_out_button_font = sign_out_button.font()
        sign_out_button_font.setPointSize(10)
        sign_out_button.setFont(sign_out_button_font)
        sign_out_button.setMaximumSize(120, 60)
        sign_out_button.setMinimumSize(120, 60)
        sign_out_button.setIcon(QIcon("./images/sign_out_btn.png"))
        sign_out_button.setIconSize(QSize(25, 25))
        sign_out_button.setCursor(QCursor(Qt.PointingHandCursor))
        sign_out_button.setStyleSheet(
            "* {background-color: #cfbaa3; border: 1px solid black;} *:hover {background-color: #ba9a75;}")
        sign_out_button.clicked.connect(self.sign_out_button_clicked)
        buttons_left_layout.addWidget(sign_out_button)

        buttons_left_widget = QWidget()
        buttons_left_widget.setLayout(buttons_left_layout)
        buttons_left_widget.setMaximumSize(4000, 60)
        buttons_layout.addWidget(buttons_left_widget)

        # Music play or pause button
        buttons_right_layout = QHBoxLayout()
        buttons_right_layout.setContentsMargins(0, 0, 0, 0)
        buttons_right_layout.setAlignment(Qt.AlignHCenter)

        # Button pause
        self.pause_button = QPushButton()
        self.pause_button.setMaximumSize(60, 60)
        self.pause_button.setMinimumSize(60, 60)
        self.pause_button.setIcon(QIcon("./images/pause_btn.png"))
        self.pause_button.setProperty("icon_name", "pause")
        self.pause_button.setIconSize(QSize(30, 30))
        self.pause_button.setStyleSheet("* {background-color: #f7c997; border: 1px solid black;} *:hover {background-color: #ffb96b;}")
        self.pause_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.pause_button.clicked.connect(self.pause_button_clicked)
        buttons_right_layout.addWidget(self.pause_button)

        buttons_right_widget = QWidget()
        buttons_right_widget.setLayout(buttons_right_layout)
        buttons_layout.addWidget(buttons_right_widget)
        buttons_right_widget.setMinimumSize(90, 60)
        buttons_right_widget.setMaximumSize(90, 60)

        buttons_widget = QWidget()
        buttons_widget.setLayout(buttons_layout)
        # buttons_widget.setMaximumSize(1100, 200)
        animation_layout.addWidget(buttons_widget)

        animation_widget = QWidget()
        animation_widget.setLayout(animation_layout)
        self.stacked_widget.addWidget(animation_widget)
        # --- End of Animation widget

        # --- Rating widget
        rating_layout = QVBoxLayout()
        rating_layout.setAlignment(Qt.AlignHCenter)

        rate_layout = QHBoxLayout()
        rate_layout.setAlignment(Qt.AlignHCenter)

        rate_label = QLabel("Rate your emotion:")
        rate_label_font = rate_label.font()
        rate_label_font.setPointSize(20)
        rate_label.setFont(rate_label_font)
        rate_layout.addWidget(rate_label)

        rate_widget = QWidget()
        rate_widget.setLayout(rate_layout)
        rate_widget.setMaximumSize(2000, 60)
        rating_layout.addWidget(rate_widget)

        # Blank space three
        blank_space_three = QLabel()
        blank_space_three.setMaximumSize(10, 30)
        rating_layout.addWidget(blank_space_three)

        # Quadrants
        quadrants_layout = QHBoxLayout()
        quadrants_layout.setAlignment(Qt.AlignHCenter)

        quadrants = QuadrantWidget()
        quadrants.setMaximumSize(400, 400)
        quadrants.setMinimumSize(400, 400)
        quadrants_layout.addWidget(quadrants)

        quadrants_widget = QWidget()
        quadrants_widget.setLayout(quadrants_layout)
        rating_layout.addWidget(quadrants_widget)

        # Submit emotion button
        submit_layout = QHBoxLayout()
        submit_layout.setAlignment(Qt.AlignHCenter)

        submit_btn = QPushButton("Submit")
        submit_font = submit_btn.font()
        submit_font.setPointSize(20)
        submit_btn.setFont(submit_font)
        submit_btn.setMinimumSize(50, 50)
        submit_btn.setIcon(QIcon('./images/tune_in_btn.png'))
        submit_btn.setIconSize(QSize(50, 50))
        submit_btn.setFlat(True)
        submit_btn.setStyleSheet("QPushButton { background-color: transparent;}")
        submit_btn.setCursor(QCursor(Qt.PointingHandCursor))
        submit_btn.clicked.connect(self.emotion_rated)
        submit_layout.addWidget(submit_btn)

        submit_widget = QWidget()
        submit_widget.setLayout(submit_layout)
        rating_layout.addWidget(submit_widget)

        # Blank space five
        blank_space_five = QLabel()
        blank_space_five.setMaximumSize(10, 800)
        rating_layout.addWidget(blank_space_five)

        rating_widget = QWidget()
        rating_widget.setLayout(rating_layout)
        self.stacked_widget.addWidget(rating_widget)
        # --- End of Rating widget

        # --- Play Next Music widget
        play_next_layout = QVBoxLayout()
        play_next_layout.setAlignment(Qt.AlignHCenter)

        if is_in_building_dataset_phase:
            # Training Progress Slider
            progress_layout_vertical = QVBoxLayout()
            progress_layout_vertical.setAlignment(Qt.AlignHCenter)

            # Slider value
            slider_line_layout = QHBoxLayout()
            slider_line_layout.setContentsMargins(0, 0, 0, 0)
            slider_line_layout.setSpacing(20)

            slider_blank_space = QLabel()
            slider_blank_space_font = slider_blank_space.font()
            slider_blank_space_font.setPointSize(15)
            slider_blank_space.setFont(slider_blank_space_font)
            slider_blank_space.setMaximumSize(100, 30)
            slider_line_layout.addWidget(slider_blank_space)

            self.slider_value_label_play_next = QLineEdit(str(current_user_bpd_progress) + "%")
            self.slider_value_label_play_next.setReadOnly(True)

            slider_font = self.slider_value_label_play_next.font()
            slider_font.setPointSize(13)
            self.slider_value_label_play_next.setFont(slider_font)

            self.slider_value_label_play_next.setStyleSheet(
                "* { background-color: rgba(0, 0, 0, 0); border: rgba(0, 0, 0, 0); z-index: 1}");
            self.slider_value_label_play_next.setMaximumSize(810, 50)
            self.slider_value_label_play_next.setMinimumSize(70, 50)

            self.slider_value_label_play_next.textChanged.connect(self.move_slider_label)
            slider_line_layout.addWidget(self.slider_value_label_play_next)

            slider_line_widget = QWidget()
            slider_line_widget.setLayout(slider_line_layout)
            slider_line_widget.setMaximumSize(1000, 30)
            slider_line_widget.setMinimumSize(1000, 30)
            progress_layout_vertical.addWidget(slider_line_widget)

            progress_line_layout = QHBoxLayout()
            progress_line_layout.setContentsMargins(0, 0, 0, 0)
            progress_line_layout.setSpacing(20)

            progress_label = QLabel("Progress")
            progress_font = progress_label.font()
            progress_font.setPointSize(15)
            progress_label.setFont(progress_font)
            progress_label.setMaximumSize(100, 60)
            progress_label.setMinimumSize(100, 60)
            progress_line_layout.addWidget(progress_label)

            self.progress_slider_play_next = QSlider(Qt.Horizontal)
            self.progress_slider_play_next.setMinimum(0)
            self.progress_slider_play_next.setValue(current_user_bpd_progress)
            self.progress_slider_play_next.setMaximum(100)
            # self.progress_slider.setSingleStep(round(100/self.music_files_length))
            self.progress_slider_play_next.setMaximumSize(800, 40)
            self.progress_slider_play_next.setStyleSheet("QSlider::groove:horizontal "
                                                         "{border: 1px solid #999999; height: 8px;"
                                                         "margin: 2px 0;} "
                                                         "QSlider::handle:horizontal "
                                                         "{background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #f7c997, stop:1 #ffffff);"
                                                         "border: 1px solid #f7c997; width: 0px;"
                                                         "margin: -5px 0; border-radius: 3px;}"
                                                         "QSlider::add-page:horizontal {background: white}"
                                                         "QSlider::sub-page:horizontal {background: #ffd7ab}")
            self.progress_slider_play_next.valueChanged.connect(self.slider_value_changed)
            self.progress_slider_play_next.setEnabled(False)
            progress_line_layout.addWidget(self.progress_slider_play_next)

            self.slider_value_label_play_next.setMinimumSize(self.progress_slider_play_next.width() + 100, 30)

            self.slider_value_label_play_next.setContentsMargins(int(current_user_bpd_progress * 7.7), 10, 0, 0)

            progress_line_widget = QWidget()
            progress_line_widget.setLayout(progress_line_layout)
            progress_line_widget.setMaximumSize(1000, 80)
            progress_line_widget.setMinimumSize(1000, 80)

            progress_layout_vertical.addWidget(progress_line_widget)
            progress_layout_vertical_widget = QWidget()
            progress_layout_vertical_widget.setMaximumSize(2000, 110)
            progress_layout_vertical_widget.setLayout(progress_layout_vertical)

            play_next_layout.addWidget(progress_layout_vertical_widget)

        # Blank space
        blank_space = QLabel()
        blank_space.setMaximumSize(10, 120)
        play_next_layout.addWidget(blank_space)

        play_layout = QHBoxLayout()
        play_layout.setAlignment(Qt.AlignHCenter)

        play_btn = QPushButton("Play next\n music")
        play_font = play_btn.font()
        play_font.setPointSize(15)
        play_btn.setFont(play_font)
        play_btn.setStyleSheet(
            "* {background-color: #f7c997; border: 1px solid black;} *:hover {background-color: #ffb96b;}")
        play_btn.clicked.connect(self.play_next_music_clicked)
        play_btn.setMaximumSize(200, 100)
        play_btn.setMinimumSize(200, 100)
        play_layout.addWidget(play_btn)

        play_widget = QWidget()
        play_widget.setLayout(play_layout)
        play_widget.setMaximumSize(2000, 120)
        play_next_layout.addWidget(play_widget)

        # Blank space five
        blank_space_five = QLabel()
        blank_space_five.setMaximumSize(10, 800)
        play_next_layout.addWidget(blank_space_five)

        play_next_widget = QWidget()
        play_next_widget.setLayout(play_next_layout)
        self.stacked_widget.addWidget(play_next_widget)
        # --- End of Play Next Music widget

        # --- Finished BDP widget
        finished_bdp_layout = QVBoxLayout()
        finished_bdp_layout.setAlignment(Qt.AlignHCenter)

        # Blank space
        blank_space = QLabel()
        blank_space.setMaximumSize(10, 60)
        finished_bdp_layout.addWidget(blank_space)

        # Congrats label
        congrats_layout = QHBoxLayout()
        congrats_layout.setAlignment(Qt.AlignHCenter)
        congrats_layout.setContentsMargins(0, 0, 0, 0)

        congrats_label = QLabel(f"Congrats, {current_user_name}!")
        congrats_font = congrats_label.font()
        congrats_font.setPointSize(20)
        congrats_label.setFont(congrats_font)
        congrats_label.setMaximumSize(200 + len(current_user_name), 70)
        congrats_label.setMinimumSize(240 + len(current_user_name), 70)
        congrats_layout.addWidget(congrats_label)

        congrats_widget = QWidget()
        congrats_widget.setLayout(congrats_layout)
        congrats_widget.setMaximumSize(2000, 70)
        finished_bdp_layout.addWidget(congrats_widget)

        # Finished label
        finished_layout = QHBoxLayout()
        finished_layout.setAlignment(Qt.AlignHCenter)
        finished_layout.setContentsMargins(0, 0, 0, 0)

        finished_label = QLabel("Finished Building Dataset Phase")
        finished_font = finished_label.font()
        finished_font.setPointSize(25)
        finished_label.setFont(finished_font)
        finished_label.setMaximumSize(600, 80)
        finished_label.setMinimumSize(600, 80)
        finished_layout.addWidget(finished_label)

        finished_widget = QWidget()
        finished_widget.setLayout(finished_layout)
        finished_widget.setMaximumSize(2000, 80)
        finished_bdp_layout.addWidget(finished_widget)

        # Continue button
        continue_layout = QHBoxLayout()
        continue_layout.setAlignment(Qt.AlignHCenter)

        continue_btn = QPushButton("Continue")
        continue_font = continue_btn.font()
        continue_font.setPointSize(18)
        continue_btn.setFont(continue_font)
        continue_btn.setStyleSheet(
            "* {background-color: #f7c997; border: 1px solid black;} *:hover {background-color: #ffb96b;}")
        continue_btn.clicked.connect(self.finished_btn_clicked)
        continue_btn.setMaximumSize(180, 80)
        continue_btn.setMinimumSize(180, 80)
        continue_layout.addWidget(continue_btn)

        continue_widget = QWidget()
        continue_widget.setLayout(continue_layout)
        continue_widget.setMaximumSize(2000, 100)
        finished_bdp_layout.addWidget(continue_widget)

        # Blank space five
        blank_space_five = QLabel()
        blank_space_five.setMaximumSize(10, 800)
        finished_bdp_layout.addWidget(blank_space_five)

        finished_bdp_widget = QWidget()
        finished_bdp_widget.setLayout(finished_bdp_layout)
        self.stacked_widget.addWidget(finished_bdp_widget)
        # --- End of Play Next Music widget

        base_layout.addWidget(self.stacked_widget)
        # self.switch_layout()

        base_widget = Color('#f5e6d0')
        base_widget.setLayout(base_layout)
        self.setCentralWidget(base_widget)

    def slider_value_changed(self, value):
        self.slider_value_label_animation.setText(str(value)+"%")
        self.slider_value_label_play_next.setText(str(value)+"%")

    def move_slider_label(self, value):
        value_number = int(strip(value.split('%')[0]).flat[0])
        self.slider_value_label_animation.setContentsMargins(int((self.slider_value_label_animation.width() * value_number)/100)-20, 13, 0, 0)
        self.slider_value_label_play_next.setContentsMargins(int((self.slider_value_label_play_next.width() * value_number)/100)-20, 13, 0, 0)

    def volume_slider_value_changed(self, value):
        self.music_thread.set_volume(value/100)

    def pause_button_clicked(self):
        if self.pause_button.property("icon_name") == "pause":
            self.music_thread.pause_music()
            self.emotion_thread.pause_emotions()
            self.music_is_paused = True
            self.pause_button.setIcon(QIcon("./images/play_btn.png"))
            self.pause_button.setProperty("icon_name", "play")
            self.pause_button.setIconSize(QSize(30, 30))
        else:
            self.music_thread.resume_music()
            self.emotion_thread.resume_emotions()
            self.music_is_paused = False
            self.pause_button.setIcon(QIcon("./images/pause_btn.png"))
            self.pause_button.setProperty("icon_name", "pause")
            self.pause_button.setIconSize(QSize(30, 30))

    def emotion_captured(self, result):
        self.initial_emotion = result
        return

    def choose_next_application_music(self):
        global valence_arousal_pairs
        global application_music_names

        # Check if camera is available
        success, frames = self.emotion_thread.video.read()
        if not success:
            QMessageBox.information(
                self, "Error", "Your camera is not properly working,\n please fix that and try again",
                QMessageBox.Ok,
            )

            return None
        else:  # Camera is available
            # 1. Capture initial emotion
            self.emotion_thread.capture_one_emotion()
            while not self.initial_emotion:
                continue

            # 2. Create percentages line
            initial_emotion_percentages = 'NA|'
            for emotion in self.initial_emotion['emotion']:
                initial_emotion_percentages += str(round(self.initial_emotion['emotion'][emotion], 3))
                if emotion != 'neutral':  # last emotion
                    initial_emotion_percentages += '-'
                else:
                    initial_emotion_percentages += '|' + self.initial_emotion['dominant_emotion']

            current_time = datetime.now().strftime("%H:%M:%S")  # gets current time
            new_dict = {'listenedAt': current_time,
                        'instant_seconds|percentages|dominant_emotion': initial_emotion_percentages,
                        'initial_emotion': self.initial_emotion['dominant_emotion']}

            # 2 - Get context
            update_time = datetime.now()
            global last_time_context_data_was_called
            global last_context_data

            # --- Update context if there is no context or if 20 minutes have elapsed since last time ---
            if last_time_context_data_was_called == "" or (
                    (update_time - last_time_context_data_was_called).total_seconds() > 1200):
                context_dictionary, number_of_headers = get_context()
                if number_of_headers == 18: # If there are 18 headers, then there was no API error from any of the APIs
                    last_time_context_data_was_called = update_time
                last_context_data = context_dictionary
            else:
                context_dictionary = last_context_data
            new_dict.update(context_dictionary)

            # 3. Get normalized dataframe
            df = pd.DataFrame(new_dict, index=[0])
            filtered_df = normalize_dataset(df)

            # 4. Predict valence and arousal
            model = keras.models.load_model(f'../MusicPredictModels/{current_user_name.lower()}_music_predict.h5')
            predictions = model.predict(filtered_df)[0]

            # Extract the predicted valence and arousal values
            predicted_valence = round(convert_to_new_range(0, 1, -1, 1, predictions[0]), 3)
            predicted_arousal = round(convert_to_new_range(0, 1, -1, 1, predictions[1]), 3)

            print("Valence: " + str(predicted_valence) + " Arousal: " + str(predicted_arousal))

            # 5. Calculate distance between predicted valence and arousal and the pair's valence and arousal
            # Create a NearestNeighbors model and fit your data
            nbrs = NearestNeighbors(n_neighbors=1).fit(valence_arousal_pairs)

            # Define your new point
            new_point = pd.DataFrame([[predicted_valence, predicted_arousal]], columns=['music_valence', 'music_arousal'])

            # Find the index of the closest point to the new point
            distance, index = nbrs.kneighbors(new_point)

            # 6. Get name of the music to play -> music with the minimum distance
            music_name = application_music_names[index[0][0]]

            return music_name

    def save_bdp_progress_to_csv(self):
        global current_user_name
        global current_user_bpd_progress
        global musics_listened_by_current_user_in_current_session

        #Check if the user didn't listen to anything
        if current_user_bpd_progress == 0:
            return

        first_write = not os.path.isfile('../users.csv')  # checks if file exists
        user_line_number = -1
        progress = -1
        every_music_listened = '' # List of musics that the user listened before the current session
        lines = []
        if not first_write:
            with open('../users.csv', 'r', encoding='utf-8') as file:
                for i, line in enumerate(file):
                    if i == 0:
                        continue # Ignore header line in the file
                    line = line.replace('\n', '') # Remove the '\n' character
                    line_content = line.split('~~~')
                    lines.append(line_content)
                    user_name = line_content[0]
                    if user_name == current_user_name.lower():
                        progress = line_content[1]
                        every_music_listened = line_content[2]
                        user_line_number = i

        delimiter = '~~~'
        for music in musics_listened_by_current_user_in_current_session:
            if every_music_listened != '':
                every_music_listened += '___'
            every_music_listened += music

        if progress and int(progress) != current_user_bpd_progress:
            if user_line_number != -1:
                # If the user already exists, then update the progress
                lines[user_line_number-1] = [current_user_name.lower(), str(current_user_bpd_progress), every_music_listened]
            else:
                lines.append([current_user_name.lower(), str(current_user_bpd_progress), every_music_listened])

            with open('../users.csv', 'w', newline='', encoding='utf-8') as f:
                header = ['USERNAME', 'DATASET_PROGRESS', 'MUSICS_LISTENED']
                h = delimiter.join(header)
                f.write(h + '\n')
                for line in lines:
                    l = delimiter.join(line)
                    f.write(l + '\n')

    def stop_threads(self):
        self.music_thread.pause_music()
        self.music_thread.exit(0)
        self.music_thread = None
        self.emotion_thread.stop_emotions()
        self.emotion_thread.exit(0)
        self.emotion_thread = None

    def save_user_progress(self):
        global data

        first_write = not os.path.isfile('../dataset_for_model_training.csv')  # checks if dataset file exists

        # If data has values, append to csv file to build the dataset
        if data:
            with open('../dataset_for_model_training.csv', 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                if first_write:
                    header_row = ['username', 'listenedAt', 'initial_emotion', 'music_name',
                                  'last_emotion', 'rated_emotion', 'instant_seconds|percentages|dominant_emotion',
                                  'precipitaProb', 'tMin', 'tMax', 'idWeatherType', 'classWindSpeed',
                                  'classPrecInt', 'sunrise', 'sunset', 'day_length', 'timeOfDay', 'isWorkDay',
                                  'cloud_pct', 'temp', 'feels_like', 'humidity', 'wind_speed']

                    writer.writerow(header_row)

                for record in data:
                    writer.writerow(record.values())

        self.save_bdp_progress_to_csv()

    def quit_button_clicked(self):
        reply = confirm_warning(self, "Confirm Exit", "You're about to leave the application.\n Are you sure?")

        if reply == QMessageBox.Yes:
            self.stop_threads()

            global is_in_building_dataset_phase
            if is_in_building_dataset_phase:
                self.save_user_progress()

            quit(0)

    def sign_out_button_clicked(self):
        reply = confirm_warning(self, "Confirm Sign Out", "You're about to sign out.\n Are you sure?")

        if reply == QMessageBox.Yes:
            self.stop_threads()

            global is_in_building_dataset_phase
            if is_in_building_dataset_phase:
                self.save_user_progress()

            global current_user_name
            current_user_name = ''
            global current_user_bpd_progress
            current_user_bpd_progress = 0

            # Switches to Login Window
            self.nextWindow = LoginWindow()
            self.nextWindow.show()
            self.close()

    def closeEvent(self, event):
        if not ("LoginWindow" in str(self.nextWindow)) and not ("ApplicationHomeScreen" in str(self.nextWindow)) and not ("TrainingModelScreen" in str(self.nextWindow)):
            reply = confirm_warning(self, "Confirm Exit", "You're about to leave the application.\n Are you sure?")
            if reply == QMessageBox.Yes:
                self.stop_threads()

                global is_in_building_dataset_phase
                if is_in_building_dataset_phase:
                    self.save_user_progress()

                quit(0)
            else:
                event.ignore()

    class PaintLine(QWidget):
        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)

            pen = QPen(Qt.SolidLine)
            pen.setColor(Qt.black)
            pen.setWidth(2)
            painter.setPen(pen)

            # Draw a diagonal line from top-left to bottom-right
            painter.drawLine(20, 0, 20, 40)

            # Draw a diagonal line from top-left to bottom-right
            painter.drawLine(30, 0, 30, 60)

    def switch_layout(self):
        if self.music_playing:
            self.stacked_widget.setCurrentIndex(0)
            self.music_thread.start()
            self.emotion_thread.start()
        else:
            if self.is_rating_music:
                self.stacked_widget.setCurrentIndex(1)
            else:
                global current_user_bpd_progress
                if current_user_bpd_progress != 100:
                    # Play next music button
                    self.stacked_widget.setCurrentIndex(2)
                else: # Finished BDP Phase
                    self.stacked_widget.setCurrentIndex(3)

    def emotion_rated(self):
        global rated_emotion
        global data
        global new_record
        global musics_listened_by_current_user
        global current_user_bpd_progress
        global current_user_name

        self.setDisabled(True)
        self.is_rating_music = False
        musics_listened_by_current_user.append(new_record['music_name'])
        musics_listened_by_current_user_in_current_session.append(new_record['music_name'])
        current_user_bpd_progress = round((len(musics_listened_by_current_user) * 100) / self.music_files_length) # Regra 3 simples para ver progresso atual
        self.progress_slider_animation.setValue(current_user_bpd_progress)
        self.progress_slider_play_next.setValue(current_user_bpd_progress)
        self.switch_layout()
        current_time = datetime.now().strftime("%H:%M:%S")  # gets current time


        # TODO - alterar new_record['instant_seconds|percentages|dominant_emotion'] consoante rated emotion
        new_dict = {'username': current_user_name.lower(), 'listenedAt': current_time, 'initial_emotion': new_record['initial_emotion'],
                    'music_name': new_record['music_name'],
                    'last_emotion': new_record['last_emotion'],
                    'rated_emotion': str(rated_emotion[0])+'|'+str(rated_emotion[1]),
                    'instant_seconds|percentages|dominant_emotion': new_record['instant_seconds|percentages|dominant_emotion']
                    }

        # Get current time
        update_time = datetime.now()
        global last_time_context_data_was_called
        global last_context_data
        # Update context if there is no context or if 20 minutes have elapsed since last time
        if last_time_context_data_was_called == "" or ((update_time - last_time_context_data_was_called).total_seconds() > 1200):
            context_dictionary, number_of_headers = get_context()
            if number_of_headers == 18:
                last_time_context_data_was_called = update_time
            last_context_data = context_dictionary
        else:
            context_dictionary = last_context_data

        new_dict.update(context_dictionary)
        data.append(new_dict)
        new_record = reset_values(new_record)

        self.setDisabled(False)

    # def angry_button_clicked(self):
    #     self.emotion_rated("angry")
    #
    # def disgust_button_clicked(self):
    #     self.emotion_rated("disgust")
    #
    # def fear_button_clicked(self):
    #     self.emotion_rated("fear")
    #
    # def sad_button_clicked(self):
    #     self.emotion_rated("sad")
    #
    # def neutral_button_clicked(self):
    #     self.emotion_rated("neutral")
    #
    # def surprise_button_clicked(self):
    #     self.emotion_rated("surprise")
    #
    # def happy_button_clicked(self):
    #     self.emotion_rated("happy")

    def play_next_music_clicked(self):
        # self.music_thread.set_new_music('Agitated Celtic music 30 seconds.mp3')
        global new_record
        # global is_in_building_dataset_phase
        # if is_in_building_dataset_phase:
        music_name = self.pick_next_music_to_play_in_BDP()

        new_record['music_name'] = music_name
        self.music_thread.start()
        self.music_playing = True
        self.switch_layout()

    def pick_next_music_to_play_in_BDP(self):
        global musics_listened_by_current_user
        global has_user_finished_first_iteration_of_bdp

        #TODO - falta fazer com que o progresso = 100 nao seja só até acabar o music files
        if not has_user_finished_first_iteration_of_bdp and len(musics_listened_by_current_user) == self.music_files_length:
            has_user_finished_first_iteration_of_bdp = True

        if has_user_finished_first_iteration_of_bdp:
            random_music = random.choice(self.music_files)
        else:
            while True:
                random_music = random.choice(self.music_files)
                if random_music not in musics_listened_by_current_user:
                    break

        music_full_path = self.music_thread.set_music(random_music)

        # Set the duration of the music using Mutagen
        # audio = MP3(self.music_thread.directory+"/"+random_music)
        audio = MP3(music_full_path)
        self.music_progress.set_duration(int(audio.info.length))

        return random_music

    def music_finished(self):
        if not self.music_is_paused:
            global is_in_building_dataset_phase
            if is_in_building_dataset_phase:
                self.music_playing = False
                if self.emotion_thread != None:
                    self.emotion_thread.stop_emotions()
                self.is_rating_music = True
                self.switch_layout()
            else:
                self.emotion_thread.stop_emotions()

                # Choose next music
                music_name = self.choose_next_application_music()

                if not music_name:
                    # TODO - mostrar erro
                    return

                music_full_path = self.music_thread.set_music(music_name)

                # Set the duration of the music using Mutagen
                audio = MP3(music_full_path)
                self.music_progress.set_duration(int(audio.info.length))

                self.music_thread.start()

    def finished_btn_clicked(self):
        global current_user_name
        global is_in_building_dataset_phase
        is_in_building_dataset_phase = False
        global is_training_model
        is_training_model = True

        # self.emotion_thread.stop_emotions()
        self.emotion_thread.quit()

        self.save_bdp_progress_to_csv()
        self.save_user_progress()


        # Get dataframe
        with open('../dataset_for_model_training.csv', 'r') as file_obj:
            dataset = pd.read_csv(file_obj)
            df = pd.DataFrame(dataset) # So that we don't change the original df (and filtered_df can't be a view)
            df = df[df['username'] == current_user_name.lower()]

            # Normalize dataframe
            filtered_df = normalize_dataset(df)

            # Save BDP normalized dataset
            filtered_df.to_csv(f'../{current_user_name.lower()}_normalized_dataset.csv', index=False)

            self.nextWindow = TrainingModelScreen()
            self.nextWindow.show()
            self.nextWindow.train_thread.start()
            self.close()


class MusicThread(QThread):
    finished_music_signal = pyqtSignal()

    def __init__(self, directory, personalized_directory, parent=None):
        super().__init__(parent)

        self.directory = directory
        self.personalized_directory = personalized_directory

        self.defined_volume = -1
        self.music_is_paused = False

    def pause_music(self):
        self.music_is_paused = True
        pygame.mixer.music.pause()

    def resume_music(self):
        self.music_is_paused = False
        pygame.mixer.music.unpause()

    def set_volume(self, volume):
        pygame.mixer.music.set_volume(volume)
        self.defined_volume = volume

    def set_music(self, music_name):
        if music_name == "NA":
            return
        global_available_musics = os.listdir(self.directory)
        music_full_path = ""
        music_found = False
        if os.path.exists(self.personalized_directory):
            personalized_available_musics = os.listdir(self.personalized_directory)
            for music in personalized_available_musics:
                if music_name == music:
                    music_full_path = self.personalized_directory + '/' + music
                    music_found = True
                    break
        if not music_found:
            for music in global_available_musics:
                if music_name == music:
                    music_full_path = self.directory + '/' + music
                    break

        self.music_name = music_full_path
        return music_full_path

    def set_directory(self, directory):
        self.directory = directory

    def run(self):
        # ---------- Initialize Pygame Mixer ----------
        pygame.mixer.init()
        # pygame.mixer.music.load(self.directory + '/' + self.music_name)
        pygame.mixer.music.load(self.music_name)

        if self.defined_volume != -1:
            self.set_volume(self.defined_volume)
        else:
            self.set_volume(0.2)
        pygame.mixer.music.play()  # plays music

        # ---------- Waits for the music to end ----------
        while pygame.mixer.music.get_busy() or self.music_is_paused:
            pygame.time.wait(100)

        # ---------- Finished Music ----------
        self.finished_music_signal.emit()

        pass


class EmotionsThread(QThread):
    captured_one_emotion = pyqtSignal(dict)

    warnings.simplefilter("error")

    def __init__(self, parent=None):
        super().__init__(parent)

        self.capture_one = False

        self.emotions_running = False
        self.emotions_paused = False
        self.video = cv2.VideoCapture(0)


    def pause_emotions(self):
        self.emotions_paused = True

    def resume_emotions(self):
        self.emotions_paused = False

    def stop_emotions(self):
        global current_music_emotions
        global new_record

        self.emotions_running = False
        self.video = None
        if len(current_music_emotions) == 0:
            return

        last_emotion = current_music_emotions.split(';')[-2].split('|')[-1]  # TODO - dá erro quando nunca se apanha uma emoção
        new_record['last_emotion'] = last_emotion
        new_record['instant_seconds|percentages|dominant_emotion'] = current_music_emotions

    def append_emotion(self, dominant_emotion, time, percentages):
        global current_music_emotions
        if dominant_emotion != 'Not Found':
            current_music_emotions += str(time) + '|' + percentages + '|' + dominant_emotion + ';'

        # ---------- Update initial emotion ----------
        if new_record['initial_emotion'] == '':
            if dominant_emotion != 'Not Found':
                new_record['initial_emotion'] = dominant_emotion

    def capture_one_emotion(self):
        self.capture_one = True
        self.run()

    def run(self):
        self.emotions_running = True

        if self.capture_one:
            result = None
            while not result or result['dominant_emotion'] == 'Not Found':
                result = capture_emotion(self.video)

            self.emotions_running = False
            self.captured_one_emotion.emit(result)
            pass


        while self.emotions_running and not self.capture_one:
            # Start emotion recognition
            music_time = 6

            if not self.emotions_paused:
                result = capture_emotion(self.video)

                # ---------- Round emotions values ----------
                percentages = ''
                for emotion in result['emotion']:
                    percentages += str(round(result['emotion'][emotion], 3))
                    if emotion != 'neutral':  # last emotion
                        percentages += '-'

                self.append_emotion(result['dominant_emotion'], music_time, percentages)

                sleep(1)
                music_time += 3

        pass


class BuildingPhaseHomeScreen(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("FeelTuneAI")
        self.setMouseTracking(True)
        self.setMinimumSize(QSize(1200, 750))
        self.nextWindow = None

        global current_user_bpd_progress

        # Base Layout
        base_layout = QVBoxLayout()
        base_layout.setContentsMargins(10, 20, 10, 10)
        base_layout.setSpacing(0)

        # Title Layout
        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)

        logo = QLabel()
        logo.setPixmap(QPixmap('./images/feeltuneAI_logo.png'))
        logo.setScaledContents(True)
        logo.setMaximumSize(70, 70)
        title_layout.addWidget(logo)

        feeltune_ai_label = QLabel('FeelTune AI')
        feeltune_ai_font = feeltune_ai_label.font()
        feeltune_ai_font.setPixelSize(30)
        feeltune_ai_label.setFont(feeltune_ai_font)
        feeltune_ai_label.setMinimumSize(960, 80)
        title_layout.addWidget(feeltune_ai_label)

        title_widget = QWidget()
        title_widget.setLayout(title_layout)
        title_widget.setMaximumSize(2000, 60)
        base_layout.addWidget(title_widget)

        # Blank space one
        blank_space_one = QLabel()
        blank_space_one.setMaximumSize(10, 30)
        base_layout.addWidget(blank_space_one)

        # Welcome message
        welcome_layout = QHBoxLayout()
        welcome_layout.setContentsMargins(0, 0, 0, 0)
        welcome_layout.setAlignment(Qt.AlignHCenter)

        welcome_label = QLabel(f"Glad you tuned in, {current_user_name}!")
        welcome_font = welcome_label.font()
        welcome_font.setPointSize(20)
        welcome_label.setFont(welcome_font)
        welcome_layout.addWidget(welcome_label)

        welcome_widget = QWidget()
        welcome_widget.setLayout(welcome_layout)
        welcome_widget.setMaximumSize(2000, 80)
        welcome_widget.setMinimumSize(400, 80)
        base_layout.addWidget(welcome_widget)

        # BDP title
        title_layout = QHBoxLayout()
        title_layout.setAlignment(Qt.AlignHCenter)
        title_layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Building Dataset Phase")
        title_font = title.font()
        title_font.setPointSize(30)
        title.setFont(title_font)
        title_layout.addWidget(title)

        title_widget = QWidget()
        title_widget.setLayout(title_layout)
        title_widget.setMaximumSize(2000, 80)
        base_layout.addWidget(title_widget)

        # Percentage complete
        percentage_layout = QHBoxLayout()
        percentage_layout.setAlignment(Qt.AlignHCenter)
        percentage_layout.setContentsMargins(0, 0, 0, 0)

        percentage = QLabel(f"{current_user_bpd_progress}% complete")
        percentage_font = percentage.font()
        percentage_font.setPointSize(22)
        percentage.setFont(percentage_font)
        percentage_layout.addWidget(percentage)

        percentage_widget = QWidget()
        percentage_widget.setLayout(percentage_layout)
        percentage_widget.setMaximumSize(2000, 80)
        percentage_widget.setMinimumSize(400, 80)
        base_layout.addWidget(percentage_widget)

        # Blank space two
        blank_space_two = QLabel()
        blank_space_two.setMaximumSize(10, 30)
        base_layout.addWidget(blank_space_two)

        # Continue button
        continue_layout = QHBoxLayout()
        continue_layout.setAlignment(Qt.AlignHCenter)
        continue_layout.setContentsMargins(0, 0, 0, 0)

        continue_btn = QPushButton("Continue")
        continue_btn_font = continue_btn.font()
        continue_btn_font.setPointSize(20)
        continue_btn.setFont(continue_btn_font)
        continue_btn.setMaximumSize(200, 80)
        continue_btn.setMinimumSize(200, 80)
        continue_btn.setStyleSheet("* {background-color: #f7c997; border: 1px solid black;} *:hover {background-color: #ffb96b;}")
        continue_btn.clicked.connect(self.continue_button_clicked)
        continue_layout.addWidget(continue_btn)

        continue_widget = QWidget()
        continue_widget.setLayout(continue_layout)
        continue_widget.setMaximumSize(2000, 100)
        base_layout.addWidget(continue_widget)

        base_layout.addWidget(blank_space_two)

        # Add music button
        add_music_layout = QHBoxLayout()
        add_music_layout.setAlignment(Qt.AlignHCenter)
        add_music_layout.setContentsMargins(0, 0, 0, 0)

        add_music_btn = QPushButton("Add music")
        add_music_btn_font = add_music_btn.font()
        add_music_btn_font.setPointSize(18)
        add_music_btn.setFont(add_music_btn_font)
        add_music_btn.setMaximumSize(200, 80)
        add_music_btn.setMinimumSize(200, 80)
        add_music_btn.setStyleSheet("* {background-color: #cfbaa3; border: 1px solid black;} *:hover {background-color: #ba9a75;}")
        add_music_btn.clicked.connect(self.add_music_button_clicked)
        add_music_layout.addWidget(add_music_btn)

        add_music_widget = QWidget()
        add_music_widget.setLayout(add_music_layout)
        add_music_widget.setMaximumSize(2000, 100)
        base_layout.addWidget(add_music_widget)

        # Blank space three
        blank_space_three = QLabel()
        blank_space_three.setMinimumSize(10, 130)
        base_layout.addWidget(blank_space_three)

        # Buttons
        buttons_left_layout = QHBoxLayout()
        buttons_left_layout.setContentsMargins(50, 0, 50, 50)
        buttons_left_layout.setSpacing(15)
        buttons_left_layout.setAlignment(Qt.AlignLeft)

        # Button Quit
        quit_button = QPushButton("Quit")
        quit_button_font = quit_button.font()
        quit_button_font.setPointSize(10)
        quit_button.setFont(quit_button_font)
        quit_button.setMaximumSize(120, 60)
        quit_button.setMinimumSize(120, 60)
        quit_button.setIcon(QIcon("./images/quit_btn.png"))
        quit_button.setIconSize(QSize(35, 35))
        quit_button.setCursor(QCursor(Qt.PointingHandCursor))
        quit_button.setStyleSheet("* {background-color: #cfbaa3; border: 1px solid black;} *:hover {background-color: #ba9a75;}")
        quit_button.clicked.connect(self.quit_button_clicked)
        buttons_left_layout.addWidget(quit_button)

        # Button Sign Out
        sign_out_button = QPushButton(" Sign Out")
        sign_out_button_font = sign_out_button.font()
        sign_out_button_font.setPointSize(10)
        sign_out_button.setFont(sign_out_button_font)
        sign_out_button.setMaximumSize(120, 60)
        sign_out_button.setMinimumSize(120, 60)
        sign_out_button.setIcon(QIcon("./images/sign_out_btn.png"))
        sign_out_button.setIconSize(QSize(25, 25))
        sign_out_button.setCursor(QCursor(Qt.PointingHandCursor))
        sign_out_button.setStyleSheet(
            "* {background-color: #cfbaa3; border: 1px solid black;} *:hover {background-color: #ba9a75;}")
        sign_out_button.clicked.connect(self.sign_out_button_clicked)
        buttons_left_layout.addWidget(sign_out_button)

        buttons_left_widget = QWidget()
        buttons_left_widget.setLayout(buttons_left_layout)
        buttons_left_widget.setMaximumSize(4000, 60)
        base_layout.addWidget(buttons_left_widget)

        # Blank space three
        blank_space_three = QLabel()
        blank_space_three.setMaximumSize(10, 800)
        base_layout.addWidget(blank_space_three)

        base_widget = Color('#f5e6d0')
        base_widget.setLayout(base_layout)
        self.setCentralWidget(base_widget)

    def continue_button_clicked(self):
        global new_record

        self.nextWindow = MusicsWindow()

        success, frames = self.nextWindow.emotion_thread.video.read()
        if not success:
            QMessageBox.information(
                self, "Error", "Your camera is not properly working,\n please fix that and try again",
                QMessageBox.Ok,
            )
        else:
            self.nextWindow.music_playing = True
            self.nextWindow.switch_layout()
            music_name = self.nextWindow.pick_next_music_to_play_in_BDP()

            new_record['music_name'] = music_name
            self.nextWindow.show()
            self.close()

    def select_mp3_file(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("MP3 Files (*.mp3)")

        if file_dialog.exec_() == QFileDialog.Accepted:
            selected_file = file_dialog.selectedFiles()[0]
            print("Selected MP3 file:", selected_file)

            # Get the duration in seconds
            audio, sr = librosa.load(selected_file)
            duration_sec = librosa.get_duration(y=audio, sr=sr)
            if duration_sec < 150:
                QMessageBox.warning(
                    self, "Error", "File needs to have at least 2 minutes and 30 seconds",
                    QMessageBox.Ok,
                )
                return None
            else:
                return selected_file
        else:
            QMessageBox.warning(
                self, "Error", "File is not a mp3 file",
                QMessageBox.Ok,
            )
            return None

    def check_if_music_was_already_classified_with_va(self, file_name):
        #check in csv to see if music is already there
        with open("../building_dataset_phase_musics_va.csv", "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            lines = list(reader)
            for index, line in reversed(list(enumerate(lines, start=1))):
                music = line[0].split("~~~")[0]
                if music == file_name:
                    return True
        return False

    def add_music_button_clicked(self):
        global current_user_name
        global is_in_building_dataset_phase
        file = self.select_mp3_file()
        self.setDisabled(True)

        if file is None:
            return

        # ---------- Uploads music ----------
        try:
            first_upload = False
            folder_name = f"../personalized_musics/{current_user_name.lower()}"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            if is_in_building_dataset_phase:
                folder_name = f"../personalized_musics/{current_user_name.lower()}/building_dataset_phase_musics"
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                    first_upload = True
            else:
                folder_name = f"../personalized_musics/{current_user_name.lower()}/application_musics"
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                    first_upload = True
            if not first_upload:
                file_name = file.split('/')[-1]
                musics = os.listdir(folder_name)
                for music in musics:
                    if music == file_name:
                        self.setDisabled(False)
                        QMessageBox.information(
                            self, "Success", "Music was previously uploaded!",
                            QMessageBox.Ok,
                        )
                        return
            shutil.copy2(file, folder_name)
            file_name = file.split('/')[-1]
            if not self.check_if_music_was_already_classified_with_va(file_name):
                predict_uploaded_music_emotions(folder_name, file_name, '../building_dataset_phase_musics_va')
            self.setDisabled(False)
            QMessageBox.information(
                self, "Success", "Music uploaded!",
                QMessageBox.Ok,
            )
            # randomizeMusicOrder()
        except Exception as e:
            self.setDisabled(False)
            QMessageBox.warning(
                self, "Error", "Error uploading music file - " + str(e),
                QMessageBox.Ok,
            )

    def quit_button_clicked(self):
        reply = confirm_warning(self, "Confirm Exit", "You're about to leave the application.\n Are you sure?")

        if reply == QMessageBox.Yes:
            quit(0)

    def sign_out_button_clicked(self):
        reply = confirm_warning(self, "Confirm Sign Out", "You're about to sign out.\n Are you sure?")

        if reply == QMessageBox.Yes:
            global current_user_name
            current_user_name = ''
            global current_user_bpd_progress
            current_user_bpd_progress = 0

            # Switches to Login Window
            self.nextWindow = LoginWindow()
            self.nextWindow.show()
            self.close()

    def closeEvent(self, event):
        if not ("LoginWindow" in str(self.nextWindow)) and not ("MusicsWindow" in str(self.nextWindow)):
            reply = confirm_warning(self, "Confirm Exit", "You're about to leave the application.\n Are you sure?")
            if reply == QMessageBox.Yes:
                quit(0)
            else:
                event.ignore()

class QuadrantWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Quadrant Graphic')
        self.setGeometry(100, 100, 500, 500)

        self.point = QRect()
        self.image = QPixmap('./images/point_icon.png')

        global is_in_building_dataset_phase
        self.is_goal_emotion = not is_in_building_dataset_phase

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.point = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.point = event.pos()

            # Limit the new position to the width and height of the widget
            width = self.width()
            height = self.height()
            self.point.setX(max(12, min(self.point.x(), width-12)))
            self.point.setY(max(12, min(self.point.y(), height-24)))

            # Calculate normalized Arousal [-1, 1]
            normalized_x = round(2 * (self.point.x() / self.width()) - 1, 3)
            if normalized_x < 0:
                normalized_x = round(normalized_x - 0.06, 3)
            else:
                if normalized_x > 0:
                    normalized_x = round(normalized_x + 0.06, 3)

            # Calculate normalized Valence [-1, 1]
            normalized_y = round(-(2 * (self.point.y() / self.height()) - 1), 3)
            if normalized_y < 0:
                normalized_y = round(normalized_y - 0.12, 3)
            else:
                if normalized_y > 0:
                    normalized_y = round(normalized_y + 0.06, 3)

            if self.is_goal_emotion:
                global goal_emotion
                goal_emotion = [normalized_x, normalized_y]
            else:
                global rated_emotion
                rated_emotion = [normalized_x, normalized_y]
                print(rated_emotion)

            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()

        # Draw axes
        painter.setPen(QPen(Qt.black, 2))
        painter.drawLine(0, height // 2, width, height // 2)
        painter.drawLine(width // 2, 0, width // 2, height)

        # Draw quadrants
        painter.setPen(QPen(Qt.black, 1, Qt.DotLine))
        painter.drawLine(width // 2, 0, width // 2, height)
        painter.drawLine(0, height // 2, width, height // 2)

        # Draw labels
        font = painter.font()
        font.setPointSize(12)
        painter.setFont(font)
        metrics = QFontMetrics(font)

        # Quadrant 1
        quadrant1_label = "Sad"
        bored_label = "Bored"
        tired_label = "Tired"
        # Quadrant 2
        sleepy_label = "Sleepy"
        quadrant2_label = "Calm"
        pleased_label = "Pleased"
        # Quadrant 3
        frustrated_label = "Frustrated"
        annoyed_label = "Annoyed"
        quadrant3_label = "Angry"
        # Quadrant 4
        aroused_label = "Aroused"
        excited_label = "Excited"
        quadrant4_label = "Happy"

        # Quadrant 1
        quadrant1_pos = QPoint(10, height - metrics.height() - 120)
        bored_pos = QPoint(35, height - metrics.height() - 45)
        tired_pos = QPoint(130, height - metrics.height())
        # Quadrant 2
        sleepy_pos = QPoint(width - metrics.width(quadrant2_label) - 130, height - metrics.height())
        quadrant2_pos = QPoint(width - metrics.width(quadrant2_label) - 35, height - metrics.height() - 45)
        pleased_pos = QPoint(width - metrics.width(quadrant2_label) - 30, height - metrics.height() - 120)
        # Quadrant 3
        frustrated_pos = QPoint(90, 30)
        annoyed_pos = QPoint(30, 85)
        quadrant3_pos = QPoint(10, 160)
        # Quadrant 4
        aroused_pos = QPoint(width - metrics.width(quadrant4_label) - 120, 30)
        excited_pos = QPoint(width - metrics.width(quadrant4_label) - 35, 85)
        quadrant4_pos = QPoint(width - metrics.width(quadrant4_label) - 10, 160)

        # Quadrant 1
        painter.drawText(quadrant1_pos, quadrant1_label)
        painter.drawText(bored_pos, bored_label)
        painter.drawText(tired_pos, tired_label)
        # Quadrant 2
        painter.drawText(sleepy_pos, sleepy_label)
        painter.drawText(quadrant2_pos, quadrant2_label)
        painter.drawText(pleased_pos, pleased_label)
        # Quadrant 3
        painter.drawText(frustrated_pos, frustrated_label)
        painter.drawText(annoyed_pos, annoyed_label)
        painter.drawText(quadrant3_pos, quadrant3_label)
        # Quadrant 4
        painter.drawText(aroused_pos, aroused_label)
        painter.drawText(excited_pos, excited_label)
        painter.drawText(quadrant4_pos, quadrant4_label)

        # Draw the image as the point
        point_size = 85
        point_rect = QRect(self.point - QPoint(point_size // 2, point_size // 2), QSize(point_size, point_size))
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawPixmap(point_rect, self.image, self.image.rect())

    def resizeEvent(self, event):
        center_x = self.width() // 2
        center_y = self.height() // 2
        self.point = QPoint(center_x, center_y)


class ApplicationHomeScreen(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("FeelTuneAI")
        self.setMouseTracking(True)
        self.setMinimumSize(QSize(1200, 750))

        global current_user_name

        # Base Layout
        base_layout = QVBoxLayout()
        base_layout.setContentsMargins(10, 20, 10, 10)
        base_layout.setSpacing(0)

        # Title Layout
        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)

        logo = QLabel()
        logo.setPixmap(QPixmap('./images/feeltuneAI_logo.png'))
        logo.setScaledContents(True)
        logo.setMaximumSize(70, 70)
        title_layout.addWidget(logo)

        feeltune_ai_label = QLabel('FeelTune AI')
        feeltune_ai_font = feeltune_ai_label.font()
        feeltune_ai_font.setPixelSize(30)
        feeltune_ai_label.setFont(feeltune_ai_font)
        feeltune_ai_label.setMinimumSize(960, 80)
        title_layout.addWidget(feeltune_ai_label)

        title_widget = QWidget()
        title_widget.setLayout(title_layout)
        title_widget.setMaximumSize(2000, 60)
        base_layout.addWidget(title_widget)

        # Blank space one
        blank_space_one = QLabel()
        blank_space_one.setMaximumSize(10, 30)
        base_layout.addWidget(blank_space_one)

        # Welcome message
        welcome_layout = QHBoxLayout()
        welcome_layout.setContentsMargins(0, 0, 0, 0)
        welcome_layout.setAlignment(Qt.AlignHCenter)

        welcome_label = QLabel(f"Glad you tuned in, {current_user_name}!")
        welcome_font = welcome_label.font()
        welcome_font.setPointSize(20)
        welcome_label.setFont(welcome_font)
        welcome_layout.addWidget(welcome_label)

        welcome_widget = QWidget()
        welcome_widget.setLayout(welcome_layout)
        welcome_widget.setMaximumSize(2000, 60)
        base_layout.addWidget(welcome_widget)

        # How do you feel message
        how_message_layout = QHBoxLayout()
        how_message_layout.setAlignment(Qt.AlignHCenter)
        how_message_layout.setContentsMargins(0, 0, 0, 0)

        how_to_feel_label = QLabel("How do you want to feel today?")
        how_to_feel_font = how_to_feel_label.font()
        how_to_feel_font.setPointSize(25)
        how_to_feel_label.setFont(how_to_feel_font)
        how_message_layout.addWidget(how_to_feel_label)

        how_message_widget = QWidget()
        how_message_widget.setLayout(how_message_layout)
        how_message_widget.setMaximumSize(2000, 80)
        base_layout.addWidget(how_message_widget)

        # Blank space two
        blank_space_two = QLabel()
        blank_space_two.setMaximumSize(10, 30)
        base_layout.addWidget(blank_space_two)

        # Quadrants
        quadrants_layout = QHBoxLayout()
        quadrants_layout.setAlignment(Qt.AlignHCenter)

        quadrants = QuadrantWidget()
        quadrants.setMaximumSize(400, 400)
        quadrants.setMinimumSize(400, 400)
        quadrants_layout.addWidget(quadrants)

        quadrants_widget = QWidget()
        quadrants_widget.setLayout(quadrants_layout)
        base_layout.addWidget(quadrants_widget)

        # Submit emotion button
        submit_layout = QHBoxLayout()
        submit_layout.setAlignment(Qt.AlignHCenter)

        submit_btn = QPushButton("Submit")
        submit_font = submit_btn.font()
        submit_font.setPointSize(20)
        submit_btn.setFont(submit_font)
        submit_btn.setMinimumSize(50, 50)
        submit_btn.setIcon(QIcon('./images/tune_in_btn.png'))
        submit_btn.setIconSize(QSize(50, 50))
        submit_btn.setFlat(True)
        submit_btn.setStyleSheet("QPushButton { background-color: transparent;}")
        submit_btn.setCursor(QCursor(Qt.PointingHandCursor))
        submit_btn.clicked.connect(self.show_next_window)
        submit_layout.addWidget(submit_btn)

        submit_widget = QWidget()
        submit_widget.setLayout(submit_layout)
        base_layout.addWidget(submit_widget)

        # Blank space three
        blank_space_three = QLabel()
        blank_space_three.setMaximumSize(10, 800)
        base_layout.addWidget(blank_space_three)

        base_widget = Color('#f5e6d0')
        base_widget.setLayout(base_layout)
        self.setCentralWidget(base_widget)

    def show_next_window(self):
        global new_record
        global goal_emotion
        global current_user_name

        self.nextWindow = MusicsWindow()
        self.nextWindow.music_playing = True
        self.nextWindow.switch_layout()

        # Get every valence and arousal pairs from ApplicationMusics
        delimiter = '~~~'
        with open('../applications_musics_va.csv', 'r', encoding="utf-8") as file_obj:
            try:
                musics_df = pd.read_csv(file_obj, sep=delimiter, engine='python')
            except pd.errors.ParserError:
                musics_df = pd.read_csv(file_obj.replace(delimiter, ','), sep=',')

        if musics_df is None:
            # TODO - mostrar erro
            return

        global valence_arousal_pairs
        global application_music_names

        valence_arousal_pairs = musics_df[['music_valence', 'music_arousal']]
        application_music_names = musics_df['music_name']

        music_name = self.nextWindow.choose_next_application_music()

        if not music_name:
            # TODO - mostrar erro
            return

        music_full_path = self.nextWindow.music_thread.set_music(music_name)

        # Set the duration of the music using Mutagen
        audio = MP3(music_full_path)
        self.nextWindow.music_progress.set_duration(int(audio.info.length))

        self.nextWindow.music_thread.start()

        self.nextWindow.show()
        self.close()



class TrainingModelScreen(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("FeelTuneAI")
        self.setMouseTracking(True)
        self.setMinimumSize(QSize(1200, 750))

        self.nextWindow = None

        # Start the training in a separate thread
        self.train_thread = TrainThread()
        self.train_thread.finished_train_signal.connect(self.train_finished)

        # Base Layout
        base_layout = QVBoxLayout()
        base_layout.setContentsMargins(10, 20, 10, 10)
        base_layout.setSpacing(0)

        # Title Layout
        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)

        logo = QLabel()
        logo.setPixmap(QPixmap('./images/feeltuneAI_logo.png'))
        logo.setScaledContents(True)
        logo.setMaximumSize(70, 70)
        title_layout.addWidget(logo)

        feeltune_ai_label = QLabel('FeelTune AI')
        feeltune_ai_font = feeltune_ai_label.font()
        feeltune_ai_font.setPixelSize(30)
        feeltune_ai_label.setFont(feeltune_ai_font)
        feeltune_ai_label.setMinimumSize(960, 80)
        title_layout.addWidget(feeltune_ai_label)

        title_widget = QWidget()
        title_widget.setLayout(title_layout)
        title_widget.setMaximumSize(2000, 60)
        base_layout.addWidget(title_widget)

        self.stacked_widget = QStackedWidget()

        # --- Training Model widget
        training_model_layout = QVBoxLayout()
        training_model_layout.setAlignment(Qt.AlignHCenter)

        # Blank space one
        blank_space_one = QLabel()
        blank_space_one.setMaximumSize(10, 200)
        training_model_layout.addWidget(blank_space_one)

        # Wait message
        wait_layout = QHBoxLayout()
        wait_layout.setContentsMargins(0, 0, 0, 0)
        wait_layout.setAlignment(Qt.AlignHCenter)

        wait_label = QLabel()
        wait_label.setText("Wait a bit, we're trying to\n \u00A0\u00A0\u00A0\u00A0comprehend you :D")
        wait_font = wait_label.font()
        wait_font.setPointSize(20)
        wait_label.setFont(wait_font)
        wait_layout.addWidget(wait_label)

        wait_widget = QWidget()
        wait_widget.setLayout(wait_layout)
        wait_widget.setMaximumSize(2000, 120)
        wait_widget.setMinimumSize(600, 120)
        training_model_layout.addWidget(wait_widget)

        # Spinner layout
        spinner_layout = QHBoxLayout()
        spinner_layout.setAlignment(Qt.AlignHCenter)
        spinner_layout.setContentsMargins(0, 0, 0, 0)

        spinner = QLabel()
        movie = QMovie("./images/spinner.gif")
        spinner.setMovie(movie)
        spinner.setMaximumSize(450, 300)
        spinner.setMinimumSize(450, 300)
        movie.start()
        spinner_layout.addWidget(spinner)

        spinner_widget = QWidget()
        spinner_widget.setLayout(spinner_layout)
        training_model_layout.addWidget(spinner_widget)

        # Blank space three
        blank_space_three = QLabel()
        blank_space_three.setMaximumSize(10, 800)
        training_model_layout.addWidget(blank_space_three)

        training_model_widget = QWidget()
        training_model_widget.setLayout(training_model_layout)
        self.stacked_widget.addWidget(training_model_widget)
        # --- End of training model widget

        # --- Finished BDP widget
        finished_bdp_layout = QVBoxLayout()
        finished_bdp_layout.setAlignment(Qt.AlignHCenter)

        # Blank space
        blank_space = QLabel()
        blank_space.setMaximumSize(10, 60)
        finished_bdp_layout.addWidget(blank_space)

        # Congrats label
        congrats_layout = QHBoxLayout()
        congrats_layout.setAlignment(Qt.AlignHCenter)
        congrats_layout.setContentsMargins(0, 0, 0, 0)

        congrats_label = QLabel(f"We're done!")
        congrats_font = congrats_label.font()
        congrats_font.setPointSize(20)
        congrats_label.setFont(congrats_font)
        congrats_label.setMaximumSize(200, 70)
        congrats_label.setMinimumSize(200, 70)
        congrats_layout.addWidget(congrats_label)

        congrats_widget = QWidget()
        congrats_widget.setLayout(congrats_layout)
        congrats_widget.setMaximumSize(2000, 70)
        finished_bdp_layout.addWidget(congrats_widget)

        # Finished label
        finished_layout = QHBoxLayout()
        finished_layout.setAlignment(Qt.AlignHCenter)
        finished_layout.setContentsMargins(0, 0, 0, 0)

        finished_label = QLabel("Finished Training")
        finished_font = finished_label.font()
        finished_font.setPointSize(25)
        finished_label.setFont(finished_font)
        finished_label.setMaximumSize(320, 80)
        finished_label.setMinimumSize(320, 80)
        finished_layout.addWidget(finished_label)

        finished_widget = QWidget()
        finished_widget.setLayout(finished_layout)
        finished_widget.setMaximumSize(2000, 80)
        finished_bdp_layout.addWidget(finished_widget)

        # Continue button
        continue_layout = QHBoxLayout()
        continue_layout.setAlignment(Qt.AlignHCenter)

        continue_btn = QPushButton("Continue")
        continue_font = continue_btn.font()
        continue_font.setPointSize(18)
        continue_btn.setFont(continue_font)
        continue_btn.setStyleSheet(
            "* {background-color: #f7c997; border: 1px solid black;} *:hover {background-color: #ffb96b;}")
        continue_btn.clicked.connect(self.finished_train_btn_clicked)
        continue_btn.setMaximumSize(180, 80)
        continue_btn.setMinimumSize(180, 80)
        continue_layout.addWidget(continue_btn)

        continue_widget = QWidget()
        continue_widget.setLayout(continue_layout)
        continue_widget.setMaximumSize(2000, 100)
        finished_bdp_layout.addWidget(continue_widget)

        # Blank space five
        blank_space_five = QLabel()
        blank_space_five.setMaximumSize(10, 800)
        finished_bdp_layout.addWidget(blank_space_five)

        finished_bdp_widget = QWidget()
        finished_bdp_widget.setLayout(finished_bdp_layout)
        self.stacked_widget.addWidget(finished_bdp_widget)
        # --- End of Finished BDP widget

        base_layout.addWidget(self.stacked_widget)
        self.switch_layout()

        base_widget = Color('#f5e6d0')
        base_widget.setLayout(base_layout)
        self.setCentralWidget(base_widget)

    def train_finished(self):
        global is_training_model
        is_training_model = False
        self.switch_layout()

    def switch_layout(self):
        global is_training_model
        if is_training_model:
            self.stacked_widget.setCurrentIndex(0)
        else:
            self.stacked_widget.setCurrentIndex(1)
        return

    def train_model(self):
        global current_user_name

        def train(x_train, y_train, learning_rate, num_units, dropout_rate, epochs, batch_size):
            print(f"Training...")
            # Define the input shape
            input_shape = (x_train.shape[1],)

            # Define the inputs
            inputs = Input(shape=input_shape)

            # Define the hidden layer with one layer and num_units neurons
            hidden_layer = Dense(num_units, activation='sigmoid')(inputs)
            hidden_layer = Dropout(dropout_rate)(hidden_layer)

            # Define the output layer with 2 neurons
            outputs = Dense(2, activation='sigmoid')(hidden_layer)

            # Create the model
            model = Model(inputs=inputs, outputs=outputs)

            # Compile the model with the desired learning rate
            optimizer = SGD(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss="mse", metrics=['mean_absolute_percentage_error'])

            # Train the model
            history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

            # Return the metric values for each epoch during training
            return history.history['mean_absolute_percentage_error'][-1], model

        def objective(trial, x_train, x_test, y_train, y_test):
            # Define the hyperparameters to be optimized
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            num_units = trial.suggest_int('num_units', 32, 512)
            dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
            batch_size = trial.suggest_int('batch_size', 8, 128)
            epochs = trial.suggest_int('epochs', 50, 200)

            # Train the model and obtain the validation metric
            _, model = train(x_train, y_train, learning_rate, num_units, dropout_rate, epochs, batch_size)

            # Evaluate the model using the test data
            metric = model.evaluate(x_test, y_test)[0]

            # Return the test metric as the objective value to be optimized (minimized)
            return metric

        with open(f'../{current_user_name}_normalized_dataset.csv', 'r') as file:
            # 1. Get normalized dataset of username
            dataset = pd.read_csv(file)

            # 2. Get labels
            y = dataset[['music_valence', 'music_arousal']]

            # 3. Get context
            x = dataset.drop(labels=['music_valence', 'music_arousal'], axis=1)

            # 4. Split the data and Train the model
            print("Train and test split...")
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

            # ANTES
            '''
            # Define the input shape
            input_shape = (X_train.shape[1],)  # Replace num_features with the actual number of input features
            # Define the inputs
            inputs = Input(shape=input_shape)
            # Define the hidden layer with one layer and 1 neuron
            hidden_layer = Dense(1, activation='sigmoid')(inputs)
            # Define the output layer with 2 neurons for valence and arousal
            outputs = Dense(2, activation='sigmoid')(hidden_layer)
            # Create the model
            model = Model(inputs=inputs, outputs=outputs)

            # Compile the model with the desired learning rate
            learning_rate = 0.1
            optimizer = SGD(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss="mse", metrics=['mean_absolute_percentage_error'])

            # Train the model
            epochs = 100
            batch_size = 16
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

            # Print the metric values for each epoch during training
            print("Train results: ")
            for metric_name, metric_values in history.history.items():
                print(metric_name + ":" + str(metric_values[-1]))

            # 5. Evaluate the model on the test data
            evaluation_results = model.evaluate(X_test, y_test)

            # Print the metric values during evaluation
            print("Evaluation results: ")
            for metric_name, metric_value in zip(model.metrics_names, evaluation_results):
                print(metric_name + ": " + str(metric_value))
            '''

            study = optuna.create_study(direction='minimize')  # or 'maximize' if optimizing accuracy
            study.optimize(
                lambda trial: objective(trial, x_train, x_test, y_train, y_test), n_trials=5)

            # Plot and save the optimization history
            fig = vis.plot_optimization_history(study)
            fig.update_layout(title=f"{current_user_name.capitalize()} Model Optimization History", yaxis_title="MAPE")

            # Create folder if it does not exist
            folder_name = f"./Optuna_History_images/"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            fig.write_image(f"./Optuna_History_images/{current_user_name.lower()}_optuna_history.png")

            # Plot and save the slice plot
            fig = vis.plot_slice(study)
            fig.update_layout(title=f"{current_user_name.capitalize()} Model Slice Plot", yaxis_title="MAPE")
            fig.write_image(f"./Optuna_History_images/{current_user_name.lower()}_optuna_slice_plot.png")

            # Get the best hyperparameters
            best_trial = study.best_trial
            best_learning_rate = best_trial.params['learning_rate']
            best_num_units = best_trial.params['num_units']
            best_dropout_rate = best_trial.params['dropout_rate']
            best_epochs = best_trial.params['epochs']
            best_batch_size = best_trial.params['batch_size']

            best_metric, best_model = train(x_train, y_train, best_learning_rate, best_num_units, best_dropout_rate,
                                            best_epochs, best_batch_size)

            # Evaluate the model using the test data
            test_metric = best_model.evaluate(x_test, y_test)[0]

            print('Best training value: {:.5f}'.format(best_metric))
            print('Best test value: {:.5f}'.format(test_metric))
            print('Best parameters: {}'.format(best_trial.params))

            # 6. Save the model

            # Create folder if it does not exist
            folder_name = f"../MusicPredictModels/"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            # Save the model
            model_file = f"../MusicPredictModels/{current_user_name.lower()}_music_predict.h5"
            best_model.save(model_file)

            global is_training_model
            is_training_model = False
            self.switch_layout()

    def finished_train_btn_clicked(self):
        self.nextWindow = ApplicationHomeScreen()
        self.nextWindow.show()
        self.close()

    def closeEvent(self, event):
        if not ("ApplicationHomeScreen" in str(self.nextWindow)):
            reply = confirm_warning(self, "Confirm Exit", "You're about to leave the application.\n Are you sure?")
            if reply == QMessageBox.Yes:
                global is_training_model
                if is_training_model:
                    # TODO - quando o user fecha mas ainda se está a treinar o modelo
                    quit(0)

                quit(0)
            else:
                event.ignore()


class TrainThread(QThread):
    finished_train_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

    def run(self):
        global current_user_name

        def train(x_train, y_train, learning_rate, num_units, dropout_rate, epochs, batch_size):
            print(f"Training...")
            # Define the input shape
            input_shape = (x_train.shape[1],)

            # Define the inputs
            inputs = Input(shape=input_shape)

            # Define the hidden layer with one layer and num_units neurons
            hidden_layer = Dense(num_units, activation='sigmoid')(inputs)
            hidden_layer = Dropout(dropout_rate)(hidden_layer)

            # Define the output layer with 2 neurons
            outputs = Dense(2, activation='sigmoid')(hidden_layer)

            # Create the model
            model = Model(inputs=inputs, outputs=outputs)

            # Compile the model with the desired learning rate
            optimizer = SGD(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss="mse", metrics=['mean_absolute_percentage_error'])

            # Train the model
            history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

            # Return the metric values for each epoch during training
            return history.history['mean_absolute_percentage_error'][-1], model

        def objective(trial, x_train, x_test, y_train, y_test):
            # Define the hyperparameters to be optimized
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            num_units = trial.suggest_int('num_units', 32, 512)
            dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
            batch_size = trial.suggest_int('batch_size', 8, 128)
            epochs = trial.suggest_int('epochs', 50, 200)

            # Train the model and obtain the validation metric
            _, model = train(x_train, y_train, learning_rate, num_units, dropout_rate, epochs, batch_size)

            # Evaluate the model using the test data
            metric = model.evaluate(x_test, y_test)[0]

            # Return the test metric as the objective value to be optimized (minimized)
            return metric

        with open(f'../{current_user_name}_normalized_dataset.csv', 'r') as file:
            # 1. Get normalized dataset of username
            dataset = pd.read_csv(file)

            # 2. Get labels
            y = dataset[['music_valence', 'music_arousal']]

            # 3. Get context
            x = dataset.drop(labels=['music_valence', 'music_arousal'], axis=1)

            # 4. Split the data and Train the model
            print("Train and test split...")
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

            study = optuna.create_study(direction='minimize')  # or 'maximize' if optimizing accuracy
            study.optimize(
                lambda trial: objective(trial, x_train, x_test, y_train, y_test), n_trials=5)

            # Plot and save the optimization history
            fig = vis.plot_optimization_history(study)
            fig.update_layout(title=f"{current_user_name.capitalize()} Model Optimization History", yaxis_title="MAPE")
            # TODO -> Create folder if it does not exist
            fig.write_image(f"./Optuna_History_images/{current_user_name.lower()}_optuna_history.png")

            # Plot and save the slice plot
            fig = vis.plot_slice(study)
            fig.update_layout(title=f"{current_user_name.capitalize()} Model Slice Plot", yaxis_title="MAPE")
            fig.write_image(f"./Optuna_History_images/{current_user_name.lower()}_optuna_slice_plot.png")

            # Get the best hyperparameters
            best_trial = study.best_trial
            best_learning_rate = best_trial.params['learning_rate']
            best_num_units = best_trial.params['num_units']
            best_dropout_rate = best_trial.params['dropout_rate']
            best_epochs = best_trial.params['epochs']
            best_batch_size = best_trial.params['batch_size']

            best_metric, best_model = train(x_train, y_train, best_learning_rate, best_num_units, best_dropout_rate,
                                            best_epochs, best_batch_size)

            # Evaluate the model using the test data
            test_metric = best_model.evaluate(x_test, y_test)[0]

            print('Best training value: {:.5f}'.format(best_metric))
            print('Best test value: {:.5f}'.format(test_metric))
            print('Best parameters: {}'.format(best_trial.params))

            # TODO -> Create folder if it does not exist
            # 6. Save the model
            model_file = f"../MusicPredictModels/{current_user_name.lower()}_music_predict.h5"
            best_model.save(model_file)


        # ---------- Finished Music ----------
        self.finished_train_signal.emit()

        pass


def main():
    # download_musics_from_csv('../bdp_musics_id.csv', '../BuildingDatasetPhaseMusics')
    # predict_music_directory_emotions('../ApplicationMusics', '../applications_musics_va')
    app = QApplication([])
    window = LoginWindow()
    # window = TrainingModelScreen()
    # window = MusicsWindow()
    # window = ApplicationHomeScreen()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()

