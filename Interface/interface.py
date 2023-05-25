import csv
import os
import shutil
import threading
from datetime import datetime
from threading import Event
from time import sleep

import cv2
import librosa

import context.main as contextMain
from EmotionRecognition.EmotionDetection import capture_emotion

import pygame as pygame
from PyQt5.QtCore import QSize, Qt, QPoint, QTimer, QRect, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QPalette, QColor, QIcon, QCursor, QPainter, QPen, QFontMetrics, QKeyEvent, QMovie
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMainWindow, QLabel, QLineEdit, QVBoxLayout, \
    QHBoxLayout, QSlider, QMessageBox, QStackedWidget, QFileDialog
from numpy.core.defchararray import strip
import random

from download_from_yt import download_musics
from predict_musics_VA import predict_music_directory_emotions, predict_uploaded_music_emotions

current_user_name = ''
is_in_building_dataset_phase = False
current_user_bpd_progress = 0

# datset for model training variables
data = []
current_music_emotions = ''
new_record = {'date': '', 'initial_emotion': '', 'music_name': '', 'last_emotion': '',
              'rated_emotion': '', 'instant_seconds|percentages|dominant_emotion': ''}
context_headers_to_dataset = ""

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


class CircleAnimation(QWidget):
    def __init__(self):
        super().__init__()

        self.circles = []
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(250)  # Add a circle every second

    def update_animation(self):
        # Add a new circle at a random position
        width = self.width()
        height = self.height()
        x = random.randint(150, width - 150)
        y = random.randint(120, height - 120)
        color = QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.circles.append((QRect(x, y, 1, 1), color))
        self.update()


    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)

        # Create a copy of the circles list
        circles_copy = self.circles[:]

        # Draw all circles
        for circle, color in circles_copy:
            pen = QPen(color)
            pen.setWidth(10)
            painter.setPen(pen)
            painter.drawEllipse(circle)

        # Remove circles that have faded away
        self.circles = [(circle, color) for circle, color in self.circles if color.alpha() > 0]

        # Adjust size and fade out existing circles
        for i, (circle, color) in enumerate(self.circles):
            alpha = color.alpha()
            alpha -= 1  # Adjust fade-out speed here
            if alpha <= 0:
                del self.circles[i]  # Remove the circle after fading out completely
            else:
                # Adjust growth speed and position here
                if circle.width() < 350:
                    size = circle.width() + 0.55
                    x = circle.x() + circle.width() // 2 - round(size) // 2
                    y = circle.y() + circle.height() // 2 - round(size) // 2
                    self.circles[i] = (QRect(x, y, round(size), round(size)), QColor(color.red(), color.green(), color.blue(), alpha))
                else:
                    size = circle.width() + 0.5
                    x = circle.x() + circle.width() // 2 - round(size) // 2
                    y = circle.y() + circle.height() // 2 - round(size) // 2
                    self.circles[i] = (QRect(x, y, round(size), round(size)), QColor(color.red(), color.green(), color.blue(), 0))
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
        progress = 0
        file_exists = os.path.isfile('users.csv')
        if file_exists:
            with open('users.csv', 'r') as f:
                data = f.readlines()
                for line in data:
                    user_name = line.split(',')[0]
                    if user_name.lower() == current_user_name.lower():
                        progress = line.split(',')[1]
                        break
        global is_in_building_dataset_phase
        if progress == 100:
            is_in_building_dataset_phase = False

        global current_user_bpd_progress
        current_user_bpd_progress = progress

        if is_in_building_dataset_phase:
            self.nextWindow = BuildingPhaseHomeScreen()
            self.nextWindow.show()
            self.close()
        else:
            self.nextWindow = ApplicationHomeScreen()
            self.nextWindow.show()
            self.close()

defined_volume = -1
stop = Event()

class MusicsWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("FeelTuneAI")
        self.setMouseTracking(True)
        self.setMinimumSize(QSize(1200, 750))

        global is_in_building_dataset_phase
        global current_user_bpd_progress

        self.slider_value = 10
        self.slider_value_initial_position = 0
        self.music_playing = True
        self.is_rating_music = False

        self.stacked_widget = QStackedWidget()

        self.music_is_paused = False

        # Music Thread Initialization
        musics_directory = ''
        music_name = ''
        if is_in_building_dataset_phase:
            musics_directory = '../BuildingDatasetPhaseMusics'
            music_name = 'Käärijä - Cha Cha Cha _ Finland 🇫🇮 _ Official Music Video _ Eurovision 2023.mp3'
        else:
            musics_directory = '../ApplicationMusics'
            music_name = 'Sad music 1 minute.mp3'
        self.music_thread = MusicThread(musics_directory, music_name)
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
            self.slider_value_label = QLineEdit(str(current_user_bpd_progress) + "%")
            self.slider_value_label.setReadOnly(True)
            slider_font = self.slider_value_label.font()
            slider_font.setPointSize(13)
            self.slider_value_label.setFont(slider_font)
            self.slider_value_label.setStyleSheet("* { background-color: rgba(0, 0, 0, 0); border: rgba(0, 0, 0, 0)}");
            self.slider_value_label.setMaximumSize(800, 30)
            self.slider_value_label.textChanged.connect(self.move_slider_label)
            progress_layout_vertical.addWidget(self.slider_value_label)

            self.progress_slider = QSlider(Qt.Horizontal)
            self.progress_slider.setMinimum(0)
            self.progress_slider.setValue(current_user_bpd_progress)  # TODO - colocar a percentagem de treino do user
            self.progress_slider.setMaximum(100)
            self.progress_slider.setSingleStep(round(100/self.music_files_length))
            self.progress_slider.setMaximumSize(800, 40)
            self.progress_slider.setStyleSheet("QSlider::groove:horizontal "
                                          "{border: 1px solid #999999; height: 8px;"
                                            "margin: 2px 0;} "
                                          "QSlider::handle:horizontal "
                                          "{background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #f7c997, stop:1 #ffffff);"
                                            "border: 1px solid #f7c997; width: 18px;"
                                            "margin: -5px 0; border-radius: 3px;}"
                                          "QSlider::add-page:horizontal {background: white}"
                                          "QSlider::sub-page:horizontal {background: #ffd7ab}")
            self.progress_slider.valueChanged.connect(self.slider_value_changed)
            self.progress_slider.setEnabled(False)

            progress_layout_vertical.addWidget(self.progress_slider)
            progress_layout_vertical_widget = QWidget()
            progress_layout_vertical_widget.setMaximumSize(2000, 80)
            progress_layout_vertical_widget.setLayout(progress_layout_vertical)

            base_layout.addWidget(progress_layout_vertical_widget)

        # --- Animation widget
        animation_layout = QVBoxLayout()
        animation_layout.setAlignment(Qt.AlignHCenter)

        # Circles animation
        circle_layout = QHBoxLayout()
        circle_layout.setAlignment(Qt.AlignHCenter)

        circle_animation = CircleAnimation()
        circle_animation.setMaximumSize(800, 400)
        circle_animation.setMinimumSize(800, 400)
        circle_layout.addWidget(circle_animation)

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
                                           "border: 1px solid #4d4d5d;  height: 5px;"
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
        buttons_layout.setAlignment(Qt.AlignRight)
        buttons_layout.setSpacing(25)

        # Button Quit
        quit_button = QPushButton("Quit")
        quit_button.setMinimumSize(120, 60)
        quit_font = quit_button.font()
        quit_font.setPixelSize(25)
        quit_button.setFont(quit_font)
        quit_button.setCursor(QCursor(Qt.PointingHandCursor))
        quit_button.setStyleSheet("* {background-color: #cfbaa3; border: 1px solid black;} *:hover {background-color: #ba9a75;}")
        quit_button.clicked.connect(self.quit_button_clicked)
        buttons_layout.addWidget(quit_button)

        # Button pause
        self.pause_button = QPushButton("Pause")
        self.pause_button.setMinimumSize(120, 60)
        stop_font = self.pause_button.font()
        stop_font.setPixelSize(25)
        self.pause_button.setFont(stop_font)
        self.pause_button.setStyleSheet("* {background-color: #f7c997; border: 1px solid black;} *:hover {background-color: #ffb96b;}")
        self.pause_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.pause_button.clicked.connect(self.pause_button_clicked)
        buttons_layout.addWidget(self.pause_button)

        buttons_widget = QWidget()
        buttons_widget.setLayout(buttons_layout)
        buttons_widget.setMaximumSize(1100, 200)
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

        # Blank space five
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

        base_layout.addWidget(self.stacked_widget)
        self.switch_layout()

        base_widget = Color('#f5e6d0')
        base_widget.setLayout(base_layout)
        self.setCentralWidget(base_widget)

    def slider_value_changed(self, value):
        self.slider_value = value
        self.slider_value_label.setText(str(value)+"%")

    def move_slider_label(self, value):
        value_number = strip(value.split('%')[0])
        self.slider_value_label.move(QPoint(190 + round(int(value_number) * 7.7), 14))

    def volume_slider_value_changed(self, value):
        self.music_thread.set_volume(value/100)

    def pause_button_clicked(self):
        if self.pause_button.text() == "Pause":
            self.music_thread.pause_music()
            self.music_is_paused = True
            self.pause_button.setText("Resume")
        else:
            self.music_thread.resume_music()
            self.music_is_paused = False
            self.pause_button.setText("Pause")

    def confirm_exit(self):
        reply = QMessageBox.warning(
            self, "Confirm Exit", "You're about to leave the application.\n Are you sure?",
            QMessageBox.Yes | QMessageBox.No,
        )
        return reply


    def quit_button_clicked(self):
        reply = self.confirm_exit()

        if reply == QMessageBox.Yes:
            self.music_thread.pause_music()
            self.music_thread.exit(0)

            global data

            first_write = not os.path.isfile('../dataset_for_model_training.csv')  # checks if dataset file exists

            # If data has values, append to csv file to build the dataset
            if data:
                with open('../dataset_for_model_training.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    if first_write:
                        header_row = ['date', 'initial_emotion', 'music_name',
                                      'last_emotion', 'rated_emotion', 'instant_seconds|percentages|dominant_emotion']

                        for attribute in context_headers_to_dataset:
                            header_row.append(attribute.rstrip('\r'))

                        writer.writerow(header_row)

                    for record in data:
                        writer.writerow(record.values())


            #TODO - colocar isto numa função pois é chamado várias vezes
            first_write = not os.path.isfile('../users.csv')  # checks if file exists
            user_line_number = -1
            progress = -1
            global current_user_name
            global current_user_bpd_progress
            lines = []
            if not first_write:
                with open('../users.csv', 'r') as f:
                    reader = csv.reader(f)
                    lines = list(reader)
                    for i, line in enumerate(lines):
                        user_name = line[0]
                        if user_name.lower() == current_user_name.lower():
                            progress = line[1]
                            user_line_number = i
                            break

            # If the current user has progress to update in the csv file
            if progress and int(progress) != current_user_bpd_progress:
                if user_line_number != -1:
                    # If the user already exists, then update the progress
                    lines[user_line_number] = [current_user_name, str(current_user_bpd_progress)]
                else:
                    lines.append([current_user_name, str(current_user_bpd_progress)])

                with open('../users.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    if first_write:
                        header = ['USERNAME', 'DATASET_PROGRESS']
                        writer.writerow(header)
                    writer.writerows(lines)
            quit()

    def closeEvent(self, event):
        response = self.confirm_exit()
        if response == QMessageBox.Yes:
            self.music_thread.pause_music()
            self.music_thread.exit(0)

            global data

            first_write = not os.path.isfile('../dataset_for_model_training.csv')  # checks if dataset file exists
            # if data has values, append to csv file to build the dataset
            if data:
                with open('../dataset_for_model_training.csv', 'a', newline='', encoding="utf-8") as file:
                    writer = csv.writer(file)
                    if first_write:
                        header_row = ['date', 'initial_emotion', 'music_name',
                                      'last_emotion', 'rated_emotion', 'instant_seconds|percentages|dominant_emotion']

                        for attribute in context_headers_to_dataset:
                            header_row.append(attribute.rstrip('\r'))

                        writer.writerow(header_row)

                    for record in data:
                        writer.writerow(record.values())

            first_write = not os.path.isfile('../users.csv')  # checks if file exists
            user_line_number = -1
            progress = -1
            global current_user_name
            global current_user_bpd_progress
            lines = []
            if not first_write:
                with open('../users.csv', 'r') as f:
                    reader = csv.reader(f)
                    lines = list(reader)
                    for i, line in enumerate(lines):
                        user_name = line[0]
                        if user_name.lower() == current_user_name.lower():
                            progress = line[1]
                            user_line_number = i
                            break

            # If the current user has progress to update in the csv file
            if progress and int(progress) != current_user_bpd_progress:
                if user_line_number != -1:
                    # If the user already exists, then update the progress
                    lines[user_line_number] = [current_user_name, str(current_user_bpd_progress)]
                else:
                    lines.append([current_user_name, str(current_user_bpd_progress)])

                with open('../users.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    if first_write:
                        header = ['USERNAME', 'DATASET_PROGRESS']
                        writer.writerow(header)
                    writer.writerows(lines)
            event.accept()
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
                self.stacked_widget.setCurrentIndex(2)

    def get_context(self):
        dataframe = contextMain.execute()
        if dataframe is not None:
            csv_context = dataframe.to_csv(index=False)
            context_headers = csv_context.split('\n')[0]
            context_headers = context_headers.split(',')

            global context_headers_to_dataset
            if len(context_headers_to_dataset) == 0:
                context_headers_to_dataset = context_headers

            context_values = csv_context.split('\n')[1]
            context_values = context_values.split(',')
            context_dict = {header: str(value).rstrip('\r') for header, value in zip(context_headers, context_values)}
            return context_dict

        return {}

    def emotion_rated(self, emotion):
        global data
        global new_record

        self.setDisabled(True)
        self.progress_slider.setValue(self.progress_slider.value() + self.progress_slider.singleStep())
        self.is_rating_music = False
        self.switch_layout()

        current_time = datetime.now().strftime("%d/%m/%YT%H:%M:%S")  # gets record data

        new_dict = {'date': current_time, 'initial_emotion': new_record['initial_emotion'],
                    'music_name': new_record['music_name'],
                    'last_emotion': new_record['last_emotion'],
                    'rated_emotion': emotion,
                    'instant_seconds|percentages|dominant_emotion': new_record['instant_seconds|percentages|dominant_emotion']
                    }


        context_dictionary = self.get_context()
        new_dict.update(context_dictionary)
        data.append(new_dict)
        new_record = reset_values(new_record)

        self.setDisabled(False)
        # TODO - quando percentagem chegar a 100% = treino concluído, colocar novo layout

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

        new_record['music_name'] = 'Käärijä - Cha Cha Cha _ Finland 🇫🇮 _ Official Music Video _ Eurovision 2023.mp3'
        self.music_thread.start()
        self.music_playing = True
        self.switch_layout()

    def music_finished(self):
        # TODO - colocar músicas de forma dinâmica, não estática
        if not self.music_is_paused:
            global is_in_building_dataset_phase
            if is_in_building_dataset_phase:
                self.music_playing = False
                self.emotion_thread.stop_emotions()
                self.is_rating_music = True
                self.switch_layout()
                # TODO - colocar outra música, de forma aleatória
            else:
                self.emotion_thread.stop_emotions()
                # self.music_thread.set_music('Calming relaxing music 30 seconds-.mp3')
                # TODO - colocar outra música, de acordo com a emoção obtida e a desejada - usar o modelo
                self.music_thread.start()

    def new_emotion(self, result):
        print(result)


class MusicThread(QThread):
    finished_music_signal = pyqtSignal()

    def __init__(self, directory, music_name, parent=None):
        super().__init__(parent)

        self.directory = directory
        self.music_name = music_name

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
        print("here")
        # ---------- Finished Music ----------
        self.finished_music_signal.emit()
        self.pause_music()

        pass


class EmotionsThread(QThread):
    new_emotion = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.emotions_running = False
        self.video = cv2.VideoCapture(0)


    def stop_emotions(self):
        global current_music_emotions
        global new_record

        self.emotions_running = False
        last_emotion = current_music_emotions.split(';')[-2].split('|')[-1]
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

        # Training title
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

        new_record['music_name'] = 'Käärijä - Cha Cha Cha _ Finland 🇫🇮 _ Official Music Video _ Eurovision 2023.mp3'

        self.nextWindow = MusicsWindow()
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

            # TODO - guardar em variáveis para posteriormente usar para emoção objetivo
            print("Y: "+str(normalized_y))
            print("X: "+str(normalized_x))

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
        self.nextWindow = MusicsWindow()
        self.nextWindow.show()
        self.close()


def main():
    # app = QApplication([])
    # download_musics(['https://www.youtube.com/watch?v=znWi3zN8Ucg', 'https://www.youtube.com/watch?v=tEwvUu1dBTs'], '../BuildingDatasetPhaseMusics')
    # predict_music_directory_emotions('../BuildingDatasetPhaseMusics', '../building_dataset_phase_musics_va')
    app = QApplication([])
    window = LoginWindow()
    # # window = MusicsWindow()
    # # window = ApplicationHomeScreen()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()

