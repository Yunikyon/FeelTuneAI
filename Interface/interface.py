import os
import threading
from threading import Event
from time import sleep

import context.main as contextMain
from EmotionRecognition.EmotionDetection import main as emotionDetectionMain, stopEmotions

import pygame as pygame
from PyQt5.QtCore import QSize, Qt, QPoint, QTimer, QRect, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QPalette, QColor, QIcon, QCursor, QPainter, QPen, QFontMetrics
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMainWindow, QLabel, QLineEdit, QVBoxLayout, \
    QHBoxLayout, QSlider, QMessageBox, QStackedWidget
from numpy.core.defchararray import strip
import random

current_user_name = ''
is_in_building_dataset_phase = True
training_percentage = 0

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
        self.circles.append((QRect(x, y, 1, 1), color))  # TODO - change for emotions
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
        global is_in_building_dataset_phase
        # TODO - atualizar is_in_training_phase de acordo com o user

        global training_percentage
        # TODO - atualizar training_percentage de acordo com o user

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

        self.slider_value = 10
        self.slider_value_initial_position = 0
        self.music_playing = False  # TODO - verificar quando a música está a tocar ou não para colocar o layout certo
        self.is_rating_music = False  # TODO
        # self.is_rating_music = QLineEdit()
        # self.is_rating_music.textChanged.connect()

        self.stacked_widget = QStackedWidget()

        # self.music_thread
        self.music_thread = MusicThread('../BuildingDatasetPhaseMusics', 'Agitated music 20 seconds.mp3')
        self.music_thread.finished_music_signal.connect(self.music_finished)

        # global stop
        # global defined_volume

        global is_in_building_dataset_phase
        global training_percentage

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

            # ---------- Threads initialization ----------
            # self.current_music_thread = self.playThread(self, '../BuildingDatasetPhaseMusics', self.music_files[0])
            # self.current_emotion_thread = self.emotionsThread()

            if self.music_files_length == 0:
                print(f"{Bcolors.WARNING} Music files length is zero" + Bcolors.ENDC)
                exit()
        else:  # Read application musics
            self.music_files_length = 1
            print("TODO")  # TODO

        if is_in_building_dataset_phase:
            # Training Progress Slider
            progress_layout_vertical = QVBoxLayout()
            progress_layout_vertical.setAlignment(Qt.AlignHCenter)

            # Slider value
            self.slider_value_label = QLineEdit(str(training_percentage)+"%")
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
            self.progress_slider.setValue(training_percentage)
            self.progress_slider.setMaximum(100)
            self.progress_slider.setSingleStep(round(100/self.music_files_length)) #TODO - dividir pelo número de músicas do dataset de treino
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
            # self.progress_slider.setEnabled(False)  #TODO - colocar sem ser comentário - só mudar o valor do slider quando acabar música e for avaliada pelo user

            progress_layout_vertical.addWidget(self.progress_slider)
            progress_layout_vertical_widget = QWidget()
            progress_layout_vertical_widget.setMaximumSize(2000, 80)
            progress_layout_vertical_widget.setLayout(progress_layout_vertical)

            base_layout.addWidget(progress_layout_vertical_widget)

        # if self.music_playing:
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

        # Stop Quit
        stop_button = QPushButton("Stop")
        stop_button.setMinimumSize(120, 60)
        stop_font = stop_button.font()
        stop_font.setPixelSize(25)
        stop_button.setFont(stop_font)
        stop_button.setStyleSheet("* {background-color: #f7c997; border: 1px solid black;} *:hover {background-color: #ffb96b;}")
        stop_button.setCursor(QCursor(Qt.PointingHandCursor))
        buttons_layout.addWidget(stop_button)

        buttons_widget = QWidget()
        buttons_widget.setLayout(buttons_layout)
        buttons_widget.setMaximumSize(1100, 200)
        animation_layout.addWidget(buttons_widget)

        animation_widget = QWidget()
        animation_widget.setLayout(animation_layout)
        self.stacked_widget.addWidget(animation_widget)  # TODO 1 Layout
        # base_layout.addWidget(buttons_widget)

        # else:  # Music finished playing
        #     if self.is_rating_music:
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
        # base_layout.addWidget(first_line_widget)

        # Blank space four
        blank_space_four = QLabel()
        blank_space_four.setMaximumSize(10, 30)
        rating_layout.addWidget(blank_space_four)
        # base_layout.addWidget(blank_space_four)

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
        surprised_button = QPushButton("Disgust")
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
        happy_button = QPushButton("Fear")
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
        # base_layout.addWidget(second_line_widget)

        # Blank space five
        blank_space_five = QLabel()
        blank_space_five.setMaximumSize(10, 800)
        rating_layout.addWidget(blank_space_five)
        # base_layout.addWidget(blank_space_five)

        rating_widget = QWidget()
        rating_widget.setLayout(rating_layout)
        self.stacked_widget.addWidget(rating_widget)

            # else:  # Button Play for next music
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
        # base_layout.addWidget(play_widget)

        # Blank space five
        blank_space_five = QLabel()
        blank_space_five.setMaximumSize(10, 800)
        play_next_layout.addWidget(blank_space_five)

        play_next_widget = QWidget()
        play_next_widget.setLayout(play_next_layout)
        self.stacked_widget.addWidget(play_next_widget)
        # base_layout.addWidget(blank_space_five)

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
        print("TODO")  #TODO
        self.music_thread.set_volume(value)

    def quit_button_clicked(self):
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Warning")
        dlg.setStandardButtons(QMessageBox.No | QMessageBox.Yes)
        dlg.setText("You're about to leave the application.\n Are you sure?")
        dlg.setIcon(QMessageBox.Warning)
        button_clicked = dlg.exec()

        if button_clicked == QMessageBox.Yes:
            quit()

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

    def angry_button_clicked(self):
        print("TODO")  # TODO

    def disgust_button_clicked(self):
        print("TODO")  # TODO

    def fear_button_clicked(self):
        print("TODO")  # TODO

    def sad_button_clicked(self):
        print("TODO")  # TODO

    def neutral_button_clicked(self):
        print("TODO")  # TODO

    def surprise_button_clicked(self):
        print("TODO")  # TODO

    def happy_button_clicked(self):
        print("TODO")  # TODO

    # def playThread(self, directory, music_name):
    #     play_thread = threading.Thread(target=self.playMusic, args=(self, directory, music_name,))
    #     return play_thread
    #
    # def emotionsThread(self):
    #     emotions_thread = threading.Thread(target=emotionDetectionMain, args=(self,))
    #     return emotions_thread

    def play_next_music_clicked(self):
        # self.music_thread.set_new_music('Agitated Celtic music 30 seconds.mp3')
        self.music_thread.start()

    def music_finished(self):
        print("Music ended")
        self.music_thread.exit(0)
        self.is_rating_music = True
        self.switch_layout()



class MusicThread(QThread):
    my_signal = pyqtSignal()
    finished_music_signal = pyqtSignal()

    def __init__(self, directory, music_name, parent=None):
        super().__init__(parent)

        # self.stop = Event()
        self.defined_volume = -1

        self.directory = directory
        self.music_name = music_name

    # def stop_music(self):
    #     self.stop.set()
        # print("RECEBIIII")

    def set_volume(self, volume):
        pygame.mixer.music.set_volume(volume)
        self.defined_volume = volume

    def set_new_music(self, music_name):
        self.music_name = music_name

    def play_music(self, directory, music_name):
        # ---------- Initialize Pygame Mixer ----------
        pygame.mixer.init()
        pygame.mixer.music.load(directory + '/' + music_name)

        if self.defined_volume != -1:
            self.set_volume(self.defined_volume)
        else:
            self.set_volume(0.04)

        # self.stop.clear()

        pygame.mixer.music.play()  # plays music

        # ---------- Waits for the music to end ----------
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)

            # ---------- If user closes program or cancel ----------
            # if self.stop.is_set():
            #     pygame.mixer.music.stop()
            #     break

        # ---------- Finished Music ----------
        # try:
        self.finished_music_signal.emit()
        # except:
        #     "ignore"  # tk error, from tkinder library we're not using

    def run(self):
        # do something here
        # ---------- Initialize Pygame Mixer ----------
        pygame.mixer.init()
        pygame.mixer.music.load(self.directory + '/' + self.music_name)

        if self.defined_volume != -1:
            self.set_volume(self.defined_volume)
        else:
            self.set_volume(0.04)

        # self.stop.clear()

        pygame.mixer.music.play()  # plays music

        # ---------- Waits for the music to end ----------
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)

            # ---------- If user closes program or cancel ----------
            # if self.stop.is_set():
            #     pygame.mixer.music.stop()
            #     break

        # ---------- Finished Music ----------
        # try:
        self.finished_music_signal.emit()
        # except:
        #     "ignore"  # tk error, from tkinder library we're not using
        pass


class BuildingPhaseHomeScreen(QMainWindow):
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

        percentage = QLabel("10% complete")
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
        self.nextWindow = MusicsWindow()
        self.nextWindow.show()
        self.close()

    def add_music_button_clicked(self):
        print("TODO") #TODO


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

        quadrant1_label = "Sad"
        quadrant2_label = "Calm"
        quadrant3_label = "Angry"
        quadrant4_label = "Happy"

        # Align labels on respective corners
        quadrant1_pos = QPoint(10, height - metrics.height() - 10)
        quadrant2_pos = QPoint(width - metrics.width(quadrant2_label) - 10, height - metrics.height() - 10)
        quadrant3_pos = QPoint(10, 20)
        quadrant4_pos = QPoint(width - metrics.width(quadrant4_label) - 10, 20)

        painter.drawText(quadrant1_pos, quadrant1_label)
        painter.drawText(quadrant2_pos, quadrant2_label)
        painter.drawText(quadrant3_pos, quadrant3_label)
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
    app = QApplication([])
    # window = LoginWindow()
    window = MusicsWindow()
    # window = ApplicationHomeScreen()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()

