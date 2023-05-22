import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QColor, QPainterPath


class WaveAnimation(QWidget):
    def __init__(self):
        super().__init__()

        self.amplitude = 10 # Adjust the wave amplitude
        self.frequency = 0.2  # Adjust the wave frequency
        self.phase = 0

        self.timer = QTimer()
        self.timer.timeout.connect(self.updateAnimation)
        self.timer.start(30)  # Adjust the animation speed

        self.wave_delay = 2  # Time delay between waves in seconds
        self.wave_count = 5  # Number of repetitive waves
        self.wave_spacing = 20  # Spacing between waves
        self.current_wave = 0
        self.timer.start(self.wave_delay * 1000)

    def updateAnimation(self):
        self.current_wave += 1
        if self.current_wave >= self.wave_count:
            self.current_wave = 0
        self.update()
        self.timer.start(self.wave_delay * 1000)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Set background color
        painter.setBrush(Qt.black)
        painter.drawRect(self.rect())

        # Set wave properties
        pen = painter.pen()
        pen.setColor(QColor(0, 255, 255))  # Set wave color
        pen.setWidth(2)
        painter.setPen(pen)

        # Draw the waves
        width = self.width()
        height = self.height()
        mid_height = height / 2

        path = QPainterPath()
        path.moveTo(0, mid_height)

        for i in range(self.wave_count):
            phase_offset = i * self.wave_spacing
            if i == self.current_wave:
                phase_offset += self.phase * self.frequency

            path = QPainterPath()
            path.moveTo(0, mid_height)

            for x in range(width):
                y = mid_height + self.amplitude * (1 + (height / 8) * (1 + 0.5 * (1 - abs((x / width * 2) - 1)))) * (1 + 0.5 * (1 - abs((x / width * 2) - 1 + self.phase * self.frequency + phase_offset)))
                path.lineTo(x, y)

            painter.drawPath(path)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Wave Animation")
        self.setGeometry(100, 100, 400, 300)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        wave_animation = WaveAnimation()
        layout.addWidget(wave_animation)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
