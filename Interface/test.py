import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QColor, QPen


class WaveAnimation(QWidget):
    def __init__(self):
        super().__init__()

        self.amplitude = 10  # Wave amplitude
        self.period = 200  # Wave period
        self.phase_shift = 0  # Wave phase shift
        self.step = 10  # Animation step size

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(100)  # Update animation every 50 milliseconds

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()
        center_y = height / 2

        pen = QPen(QColor(0, 0, 255))
        pen.setWidth(2)
        painter.setPen(pen)

        for x in range(width):
            # Calculate the normalized x-coordinate within the wave period
            t = (x + self.phase_shift) % self.period / self.period
            y = center_y + self.amplitude * (2 * self.normalized_sine(t) - 1)
            painter.drawPoint(round(x), round(y))

    def normalized_sine(self, value):
        return (1 + (1 / 2) * (2 * value - 1) + (1 / 2) * (2 * value - 1) ** 3) / 2

    def update_animation(self):
        print("update")
        self.phase_shift += self.step
        self.update()
        print("updated")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Ocean Wave Animation')
        self.setGeometry(100, 100, 800, 400)
        self.setCentralWidget(WaveAnimation())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
