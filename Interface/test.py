from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QSlider
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCursor


class SliderWithCustomCursor(QSlider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.setCursor(QCursor(Qt.PointingHandCursor))


class SliderExample(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()

        self.slider = SliderWithCustomCursor(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)

        layout.addWidget(self.slider)
        self.setLayout(layout)


if __name__ == '__main__':
    app = QApplication([])
    window = SliderExample()
    window.show()
    app.exec_()
