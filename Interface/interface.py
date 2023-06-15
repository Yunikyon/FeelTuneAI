import csv
import math
import os
import shutil
import threading
import warnings
from datetime import datetime
from threading import Event
from time import sleep

import cv2
import librosa
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from mutagen.mp3 import MP3

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
new_record = {'date': '', 'initial_emotion': '', 'music_name': '', 'last_emotion': '',
              'rated_emotion': '', 'instant_seconds|percentages|dominant_emotion': ''}

goal_emotion = None


def reset_values(record):
    record['initial_emotion'] = ''
    record['music_name'] = ''
    record['average_emotion'] = ''
    record['rated_emotion'] = ''
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

        global is_in_building_dataset_phase
        global current_user_bpd_progress

        self.slider_value = 10
        self.slider_value_initial_position = 0
        self.music_playing = False
        self.is_rating_music = False

        self.stacked_widget = QStackedWidget()

        self.music_is_paused = False

        # Music Thread Initialization
        if is_in_building_dataset_phase:
            musics_directory = '../BuildingDatasetPhaseMusics'
            # music_name = 'Käärijä - Cha Cha Cha _ Finland 🇫🇮 _ Official Music Video _ Eurovision 2023.mp3'
            # music_name = 'Mahmood - Soldi - Italy 🇮🇹 - Official Music Video - Eurovision 2019.mp3'
        else:
            musics_directory = '../ApplicationMusics'
            #TODO - é recolher contexto, emoção atual da pessoa,
            # emoção desejada da pessoa para escolher a música, baseado no treino da rede neuronal :)

            music_name = 'Sad music 1 minute.mp3'
        self.music_thread = MusicThread(musics_directory)
        self.music_thread.set_music("NA")
        self.music_thread.finished_music_signal.connect(self.music_finished)

        # Emotion Thread Initialization
        self.emotion_thread = EmotionsThread()
        self.emotion_thread.new_emotion.connect(self.new_emotion)

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

        self.music_files = []
        self.music_files_length = 0
        # Read training musics
        if is_in_building_dataset_phase:
            self.music_files = os.listdir('../BuildingDatasetPhaseMusics')
            self.music_files_length = len(self.music_files)
            global music_files_bdp_length
            music_files_bdp_length = self.music_files_length

            if self.music_files_length == 0:
                print(f"{Bcolors.WARNING} BDP music files length is zero" + Bcolors.ENDC)
                exit()
        else:  # Read application musics
            self.music_files = os.listdir('../ApplicationMusics')
            self.music_files_length = len(self.music_files)

            if self.music_files_length == 0:
                print(f"{Bcolors.WARNING} Application music files length is zero" + Bcolors.ENDC)
                exit()

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

            self.slider_value_label = QLineEdit(str(current_user_bpd_progress) + "%")
            self.slider_value_label.setReadOnly(True)

            slider_font = self.slider_value_label.font()
            slider_font.setPointSize(13)
            self.slider_value_label.setFont(slider_font)

            self.slider_value_label.setStyleSheet("* { background-color: rgba(0, 0, 0, 0); border: rgba(0, 0, 0, 0); z-index: 1}");
            self.slider_value_label.setMaximumSize(800, 30)

            self.slider_value_label.textChanged.connect(self.move_slider_label)
            slider_line_layout.addWidget(self.slider_value_label)

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
            progress_line_layout.addWidget(progress_label)


            self.progress_slider = QSlider(Qt.Horizontal)
            self.progress_slider.setMinimum(0)
            self.progress_slider.setValue(current_user_bpd_progress)
            self.progress_slider.setMaximum(100)
            self.progress_slider.setSingleStep(round(100/self.music_files_length))
            self.progress_slider.setMaximumSize(800, 40)
            self.progress_slider.setStyleSheet("QSlider::groove:horizontal "
                                          "{border: 1px solid #999999; height: 8px;"
                                            "margin: 2px 0;} "
                                          "QSlider::handle:horizontal "
                                          "{background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #f7c997, stop:1 #ffffff);"
                                            "border: 1px solid #f7c997; width: 0px;"
                                            "margin: -5px 0; border-radius: 3px;}"
                                          "QSlider::add-page:horizontal {background: white}"
                                          "QSlider::sub-page:horizontal {background: #ffd7ab}")
            self.progress_slider.valueChanged.connect(self.slider_value_changed)
            self.progress_slider.setEnabled(False)
            progress_line_layout.addWidget(self.progress_slider)

            self.slider_value_label.setMinimumSize(self.progress_slider.width(), 30)
            self.move_slider_label(str(current_user_bpd_progress)+"%")


            progress_line_widget = QWidget()
            progress_line_widget.setLayout(progress_line_layout)
            progress_line_widget.setMaximumSize(1000, 80)
            progress_line_widget.setMinimumSize(1000, 80)

            progress_layout_vertical.addWidget(progress_line_widget)
            progress_layout_vertical_widget = QWidget()
            progress_layout_vertical_widget.setMaximumSize(2000, 110)
            progress_layout_vertical_widget.setLayout(progress_layout_vertical)

            base_layout.addWidget(progress_layout_vertical_widget)

        # --- Animation widget
        animation_layout = QVBoxLayout()
        animation_layout.setAlignment(Qt.AlignHCenter)

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
        quit_button.setMaximumSize(135, 60)
        quit_button.setMinimumSize(135, 60)
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
        sign_out_button.setMaximumSize(145, 60)
        sign_out_button.setMinimumSize(145, 60)
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


        # buttons_layout.addWidget(quit_button)
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

        # Blank space two
        blank_space_two = QLabel()
        blank_space_two.setMaximumSize(10, 30)
        # base_layout.addWidget(blank_space_two)
        rating_layout.addWidget(blank_space_two)

        rate_layout = QHBoxLayout()
        rate_layout.setAlignment(Qt.AlignHCenter)

        rate_label = QLabel("Rate your emotion:")
        rate_label_font = rate_label.font()
        rate_label_font.setPointSize(20)
        rate_label.setFont(rate_label_font)
        rate_layout.addWidget(rate_label)

        rate_widget = QWidget()
        rate_widget.setLayout(rate_layout)
        rate_widget.setMaximumSize(2000, 80)
        rating_layout.addWidget(rate_widget)
        # base_layout.addWidget(rate_widget)

        # Blank space three
        blank_space_three = QLabel()
        blank_space_three.setMaximumSize(10, 30)
        rating_layout.addWidget(blank_space_three)
        # base_layout.addWidget(blank_space_three)

        # First Line of buttons
        first_line_layout = QHBoxLayout()
        first_line_layout.setAlignment(Qt.AlignHCenter)
        first_line_layout.setSpacing(30)
        first_line_layout.setContentsMargins(0, 0, 0, 0)

        # Angry button
        angry_button = QPushButton("Angry")
        angry_button.setMinimumSize(150, 80)
        angry_font = angry_button.font()
        angry_font.setPixelSize(25)
        angry_button.setFont(angry_font)
        angry_button.setCursor(QCursor(Qt.PointingHandCursor))
        angry_button.setStyleSheet(
            "* {background-color: #f7c997; border: 1px solid black;} *:hover {background-color: #ffb96b;}")
        angry_button.clicked.connect(self.angry_button_clicked)
        first_line_layout.addWidget(angry_button)

        # Disgust button
        disgust_button = QPushButton("Disgust")
        disgust_button.setMinimumSize(150, 80)
        disgust_font = disgust_button.font()
        disgust_font.setPixelSize(25)
        disgust_button.setFont(disgust_font)
        disgust_button.setCursor(QCursor(Qt.PointingHandCursor))
        disgust_button.setStyleSheet(
            "* {background-color: #f7c997; border: 1px solid black;} *:hover {background-color: #ffb96b;}")
        disgust_button.clicked.connect(self.disgust_button_clicked)
        first_line_layout.addWidget(disgust_button)

        # Fear button
        fear_button = QPushButton("Fear")
        fear_button.setMinimumSize(150, 80)
        fear_font = fear_button.font()
        fear_font.setPixelSize(25)
        fear_button.setFont(fear_font)
        fear_button.setCursor(QCursor(Qt.PointingHandCursor))
        fear_button.setStyleSheet(
            "* {background-color: #f7c997; border: 1px solid black;} *:hover {background-color: #ffb96b;}")
        fear_button.clicked.connect(self.fear_button_clicked)
        first_line_layout.addWidget(fear_button)

        # Sad button
        sad_button = QPushButton("Sad")
        sad_button.setMinimumSize(150, 80)
        sad_font = sad_button.font()
        sad_font.setPixelSize(25)
        sad_button.setFont(sad_font)
        sad_button.setCursor(QCursor(Qt.PointingHandCursor))
        sad_button.setStyleSheet(
            "* {background-color: #f7c997; border: 1px solid black;} *:hover {background-color: #ffb96b;}")
        sad_button.clicked.connect(self.sad_button_clicked)
        first_line_layout.addWidget(sad_button)

        first_line_widget = QWidget()
        first_line_widget.setLayout(first_line_layout)
        first_line_widget.setMaximumSize(2000, 80)
        rating_layout.addWidget(first_line_widget)

        # Blank space four
        blank_space_four = QLabel()
        blank_space_four.setMaximumSize(10, 30)
        rating_layout.addWidget(blank_space_four)

        # Second Line of buttons
        second_line_layout = QHBoxLayout()
        second_line_layout.setAlignment(Qt.AlignHCenter)
        second_line_layout.setSpacing(30)
        second_line_layout.setContentsMargins(0, 0, 0, 0)

        # Neutral button
        neutral_button = QPushButton("Neutral")
        neutral_button.setMinimumSize(150, 80)
        neutral_font = neutral_button.font()
        neutral_font.setPixelSize(25)
        neutral_button.setFont(neutral_font)
        neutral_button.setCursor(QCursor(Qt.PointingHandCursor))
        neutral_button.setStyleSheet(
            "* {background-color: #f7c997; border: 1px solid black;} *:hover {background-color: #ffb96b;}")
        neutral_button.clicked.connect(self.neutral_button_clicked)
        second_line_layout.addWidget(neutral_button)

        # Surprised button
        surprised_button = QPushButton("Surprised")
        surprised_button.setMinimumSize(150, 80)
        surprised_font = surprised_button.font()
        surprised_font.setPixelSize(25)
        surprised_button.setFont(surprised_font)
        surprised_button.setCursor(QCursor(Qt.PointingHandCursor))
        surprised_button.setStyleSheet(
            "* {background-color: #f7c997; border: 1px solid black;} *:hover {background-color: #ffb96b;}")
        surprised_button.clicked.connect(self.surprise_button_clicked)
        second_line_layout.addWidget(surprised_button)

        # Happy button
        happy_button = QPushButton("Happy")
        happy_button.setMinimumSize(150, 80)
        happy_font = happy_button.font()
        happy_font.setPixelSize(25)
        happy_button.setFont(happy_font)
        happy_button.setCursor(QCursor(Qt.PointingHandCursor))
        happy_button.setStyleSheet(
            "* {background-color: #f7c997; border: 1px solid black;} *:hover {background-color: #ffb96b;}")
        happy_button.clicked.connect(self.happy_button_clicked)
        second_line_layout.addWidget(happy_button)

        second_line_widget = QWidget()
        second_line_widget.setLayout(second_line_layout)
        second_line_widget.setMaximumSize(2000, 80)
        rating_layout.addWidget(second_line_widget)

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
        self.switch_layout()

        base_widget = Color('#f5e6d0')
        base_widget.setLayout(base_layout)
        self.setCentralWidget(base_widget)

    def slider_value_changed(self, value):
        self.slider_value = value
        self.slider_value_label.setText(str(value)+"%")
    def move_slider_label(self, value):
        value_number = int(strip(value.split('%')[0]).flat[0])
        self.slider_value_label.setContentsMargins(int((self.slider_value_label.width() * value_number)/100)-20, 13, 0, 0)

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

    def confirm_warning(self, title, message):
        reply = QMessageBox.warning(
            self, title, message,
            QMessageBox.Yes | QMessageBox.No,
        )
        return reply

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
                                  'cloud_pct', 'temp', 'feels_like', 'humidity', 'min_temp', 'max_temp', 'wind_speed']

                    writer.writerow(header_row)

                for record in data:
                    writer.writerow(record.values())

        self.save_bdp_progress_to_csv()

    def quit_button_clicked(self):
        reply = self.confirm_warning("Confirm Exit", "You're about to leave the application.\n Are you sure?")

        if reply == QMessageBox.Yes:
            self.stop_threads()

            global is_in_building_dataset_phase
            if is_in_building_dataset_phase:
                self.save_user_progress()

            quit(0)

    def sign_out_button_clicked(self):
        reply = self.confirm_warning("Confirm Sign Out", "You're about to sign out.\n Are you sure?")

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
        if not ("LoginWindow" in str(self.nextWindow)) and not ("ApplicationHomeScreen" in str(self.nextWindow)):
            reply = self.confirm_warning("Confirm Exit", "You're about to leave the application.\n Are you sure?")
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

    def get_context(self):
        dataframe = contextMain.execute()
        if dataframe is not None:
            csv_context = dataframe.to_csv(index=False)
            context_headers = csv_context.split('\n')[0]
            context_headers = context_headers.split(',')

            number_of_headers = len(context_headers)
            context_values = csv_context.split('\n')[1]
            context_values = context_values.split(',')
            context_dict = {header: str(value).rstrip('\r') for header, value in zip(context_headers, context_values)}
            return context_dict, number_of_headers
        return {}, 0

    def emotion_rated(self, emotion):
        global data
        global new_record
        global musics_listened_by_current_user
        global current_user_bpd_progress
        global current_user_name

        self.setDisabled(True)
        self.progress_slider.setValue(self.progress_slider.value() + self.progress_slider.singleStep())
        self.is_rating_music = False
        musics_listened_by_current_user.append(new_record['music_name'])
        musics_listened_by_current_user_in_current_session.append(new_record['music_name'])
        current_user_bpd_progress = round((len(musics_listened_by_current_user) * 100) / self.music_files_length) # Regra 3 simples para ver progresso atual
        self.switch_layout()
        current_time = datetime.now().strftime("%H:%M:%S")  # gets current time

        new_dict = {'username': current_user_name.lower(), 'listenedAt': current_time, 'initial_emotion': new_record['initial_emotion'],
                    'music_name': new_record['music_name'],
                    'last_emotion': new_record['last_emotion'],
                    'rated_emotion': emotion,
                    'instant_seconds|percentages|dominant_emotion': new_record['instant_seconds|percentages|dominant_emotion']
                    }

        # Get current time
        update_time = datetime.now()
        global last_time_context_data_was_called
        global last_context_data
        # Update context if there is no context or if 20 minutes have elapsed since last time
        if last_time_context_data_was_called == "" or ((update_time - last_time_context_data_was_called).total_seconds() > 1200):
            context_dictionary, number_of_headers = self.get_context()
            if number_of_headers == 18:
                last_time_context_data_was_called = update_time
            last_context_data = context_dictionary
        else :
            context_dictionary = last_context_data

        new_dict.update(context_dictionary)
        data.append(new_dict)
        new_record = reset_values(new_record)

        self.setDisabled(False)

    def angry_button_clicked(self):
        self.emotion_rated("angry")

    def disgust_button_clicked(self):
        self.emotion_rated("disgust")

    def fear_button_clicked(self):
        self.emotion_rated("fear")

    def sad_button_clicked(self):
        self.emotion_rated("sad")

    def neutral_button_clicked(self):
        self.emotion_rated("neutral")

    def surprise_button_clicked(self):
        self.emotion_rated("surprise")

    def happy_button_clicked(self):
        self.emotion_rated("happy")

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

        self.music_thread.set_music(random_music)

        # Set the duration of the music using Mutagen
        audio = MP3(self.music_thread.directory+"/"+random_music)
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
                self.music_thread.start()

    def new_emotion(self, result):
        print(result)

    def finished_btn_clicked(self):
        global current_user_name
        global is_in_building_dataset_phase
        is_in_building_dataset_phase = False

        self.save_bdp_progress_to_csv()
        self.save_user_progress()

        with open('../dataset_for_model_training.csv', 'r') as file_obj:

            df = pd.read_csv(file_obj)
            filtered_df = pd.DataFrame(df)
            # filtered_df = filtered_df[filtered_df['username'] == current_user_name]

        filtered_df = filtered_df.drop(labels=['username'], axis=1)
        if filtered_df is None:
            #TODO - mostrar erro
            return

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

        # def normalize_value_hour(value):
        #     min_value = 0
        #     max_value = 1440
        #     return (value - min_value) / (max_value - min_value)

        mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

        def min_max_normalization(value, min, max):
            return (value - min) / (max - min)

        # --------------- Create a new normalized dataframe for model training ---------------
        # --- Columns: listenedAt, sunset, sunrise ---
        hours_columns = ['listenedAt', 'sunset', 'sunrise', 'day_length']
        # hours_scaler = MinMaxScaler(feature_range=(0, 1))
        # filtered_df[hours_columns] = hours_scaler.fit_transform(filtered_df[hours_columns])
        for column in hours_columns:
            filtered_df[column] = filtered_df[column].apply(convert_hour_to_minutes)
            filtered_df[column] = mean_imputer.fit_transform(filtered_df[column].array.reshape(-1, 1))
            min_value = 0 # filtered_df[column].min()
            max_value = 1440 # filtered_df[column].max()
            filtered_df[column] = (filtered_df[column] - min_value) / (max_value - min_value) # filtered_df[column].apply(normalize_value_hour) #

        # --- Columns: initial_emotion, last_emotion, rated_emotion, idWeatherType, classWindSpeed, classPrecInt, timeOfDay ---
        # One Hot Encoding for categorical variables
        categorical_columns = ['initial_emotion', 'last_emotion',
                                'rated_emotion', 'idWeatherType',
                                'classWindSpeed', 'classPrecInt', 'timeOfDay', 'isWorkDay']
        mode_imputer.fit(filtered_df[categorical_columns])
        filtered_df[categorical_columns] = mode_imputer.transform(filtered_df[categorical_columns])
        categorical_columns.remove('isWorkDay') # isWorkDay is already binary
        filtered_df = pd.get_dummies(filtered_df, columns=categorical_columns)

        # Replacing missing values
        numerical_columns = ['tMin', 'tMax', 'temp', 'feels_like',
                             'min_temp', 'max_temp', 'cloud_pct',
                             'humidity', 'wind_speed', 'precipitaProb']
        mean_imputer.fit(filtered_df[numerical_columns])  # finds the mean of every column
        filtered_df[numerical_columns] = mean_imputer.transform(
            filtered_df[numerical_columns])  # replaces the missing values with the mean

        # def normalize_temperature(value):
        #     min_value = -20
        #     max_value = 50
        #     return (value - min_value) / (max_value - min_value)

        # --- Columns: tMin, tMax, temp, feels_like, min_temp, max_temp ---
        temperature_columns = ['tMin', 'tMax', 'temp', 'feels_like', 'min_temp', 'max_temp']
        for column in temperature_columns:
            min_value = -20 # filtered_df[column].min()
            max_value = 50 # filtered_df[column].max()
            # filtered_df[column] = filtered_df[column].apply(normalize_temperature)
            filtered_df[column] = (filtered_df[column] - min_value) / (max_value - min_value)
        # temp_scaler = MinMaxScaler(feature_range=(0, 1))
        # filtered_df[temperature_columns] = temp_scaler.fit_transform(filtered_df[temperature_columns])

        # def normalize_probability(value):
        #     min_value = 0
        #     max_value = 100
        #     return (value - min_value) / (max_value - min_value)

        from_0_to_100_values = ['cloud_pct', 'precipitaProb', 'humidity']
        for column in from_0_to_100_values:
            min_value = 0
            max_value = 100
            filtered_df[column] = (filtered_df[column] - min_value) / (max_value - min_value)
            # filtered_df[column] = filtered_df[column].apply(normalize_probability)

        # percentage_scaler = MinMaxScaler(feature_range=(0, 1))
        # filtered_df[from_0_to_100_values] = percentage_scaler.fit_transform(filtered_df[from_0_to_100_values])

        # --- Column: wind_speed ---
        # wind_speed_scaler = MinMaxScaler(feature_range=(0, 1))
        # filtered_df['wind_speed'] = wind_speed_scaler.fit_transform(filtered_df[['wind_speed']])

        for index, row in filtered_df.iterrows():
            filtered_df.at[index, 'wind_speed'] = min_max_normalization(row['wind_speed'], 0, 75)

        # Apply scaler
        # scaler = StandardScaler()
        # filtered_df[numerical_columns] = scaler.fit_transform(filtered_df[numerical_columns])

        # --- Column: isWorkDay ---
        filtered_df['isWorkDay'] = filtered_df['isWorkDay'].map({"Yes": 1, "No": 0})

        #TODO - save as csv
        #TODO - train model - mostrar noutro ecrã

        self.nextWindow = TrainingModelScreen()
        self.nextWindow.show()
        self.close()


class MusicThread(QThread):
    finished_music_signal = pyqtSignal()

    def __init__(self, directory, parent=None):
        super().__init__(parent)

        self.directory = directory
        # self.music_name = music_name

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
        self.music_name = music_name


        
    def set_directory(self, directory):
        self.directory = directory

    def run(self):
        # ---------- Initialize Pygame Mixer ----------
        pygame.mixer.init()
        pygame.mixer.music.load(self.directory + '/' + self.music_name)

        if self.defined_volume != -1:
            self.set_volume(self.defined_volume)
        else:
            self.set_volume(0.2)
        pygame.mixer.music.play()  # plays music

        # ---------- Waits for the music to end ---------- # TODO - descomentar para versão final, só queremos testar agora com 30 segundos
        # while pygame.mixer.music.get_busy() or self.music_is_paused:
        #     pygame.time.wait(100)

        pygame.time.wait(30000)

        # ---------- Finished Music ----------
        self.finished_music_signal.emit()
        self.pause_music()

        pass


class EmotionsThread(QThread):
    new_emotion = pyqtSignal()

    warnings.simplefilter("error")

    def __init__(self, parent=None):
        super().__init__(parent)

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

    def run(self):
        self.emotions_running = True

        # Start emotion recognition
        music_time = 6

        while self.emotions_running:
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
        welcome_widget.setMaximumSize(2000, 60)
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

    def add_music_button_clicked(self):
        file = self.select_mp3_file()
        self.setDisabled(True)

        if file is None:
            return

        # ---------- Uploads music ----------
        try:
            folder_name = "../BuildingDatasetPhaseMusics"
            shutil.copy2(file, folder_name)
            predict_uploaded_music_emotions(folder_name, file.split('/')[-1],'../building_dataset_phase_musics_va')
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


class QuadrantWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Quadrant Graphic')
        self.setGeometry(100, 100, 500, 500)

        self.point = QRect()
        self.image = QPixmap('./images/point_icon.png')

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

            global goal_emotion
            goal_emotion = [normalized_x, normalized_y]

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

            # TODO - com o algoritmo treinado
            music_name = self.nextWindow.pick_next_music_to_play_in_BDP()

            new_record['music_name'] = music_name
            self.nextWindow.show()
            self.close()


class TrainingModelScreen(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("FeelTuneAI")
        self.setMouseTracking(True)
        self.setMinimumSize(QSize(1200, 750))

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
        blank_space_one.setMaximumSize(10, 800)
        base_layout.addWidget(blank_space_one)

        # Wait message
        wait_layout = QHBoxLayout()
        wait_layout.setContentsMargins(0, 0, 0, 0)
        wait_layout.setAlignment(Qt.AlignHCenter)

        wait_label = QLabel("Wait a bit,\nwe're building your model")
        wait_font = wait_label.font()
        wait_font.setPointSize(20)
        wait_label.setFont(wait_font)
        wait_layout.addWidget(wait_label)

        wait_widget = QWidget()
        wait_widget.setLayout(wait_layout)
        wait_widget.setMaximumSize(2000, 60)
        base_layout.addWidget(wait_widget)

        # Blank space three
        blank_space_three = QLabel()
        blank_space_three.setMaximumSize(10, 800)
        base_layout.addWidget(blank_space_three)

        base_widget = Color('#f5e6d0')
        base_widget.setLayout(base_layout)
        self.setCentralWidget(base_widget)

def main():
    # download_musics_from_csv('../bdp_musics_id.csv', '../BuildingDatasetPhaseMusics')
    # predict_music_directory_emotions('../BuildingDatasetPhaseMusics', '../building_dataset_phase_musics_va')
    app = QApplication([])
    window = LoginWindow()
    # window = MusicsWindow()
    # window = ApplicationHomeScreen()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()

