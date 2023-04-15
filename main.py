import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QPlainTextEdit
from PyQt5.QtWidgets import QSizePolicy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
from knn import main


class KNNClassifierUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('kNN Classifier')

        layout = QHBoxLayout()

        # Left panel
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)

        self.filename_label = QLabel('Filename:')
        self.filename_input = QLineEdit('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
        self.split_label = QLabel('Split Ratio:')
        self.split_input = QLineEdit('0.67')
        self.k_label = QLabel('Number of Neighbors (k):')
        self.k_input = QLineEdit('3')

        self.classify_button = QPushButton('Start Classification')
        self.classify_button.clicked.connect(self.classify)
        
        left_panel.addWidget(self.filename_label)
        left_panel.addWidget(self.filename_input)
        left_panel.addWidget(self.split_label)
        left_panel.addWidget(self.split_input)
        left_panel.addWidget(self.k_label)
        left_panel.addWidget(self.k_input)
        left_panel.addWidget(self.classify_button)

        # Right panel
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout.addLayout(left_panel)
        layout.addWidget(self.canvas)
         # adjust the vertical space between widgets


        self.setLayout(layout)

    def classify(self):
        filename = self.filename_input.text()
        split = float(self.split_input.text())
        k = int(self.k_input.text())

        X_train, y_train, X_test, y_test = main(filename, split, k)
        self.plot_data(X_train, y_train, X_test, y_test)
        self.canvas.draw()

    def plot_data(self, X_train, y_train, X_test, y_test):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        y_train = np.where(y_train == 'Iris-setosa', 0, 1)
        y_test = np.where(y_test == 'Iris-setosa', 0, 1)
        
        ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color='red', marker='o', label='class 0')
        ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='blue', marker='x', label='class 1')
        ax.scatter(X_test[:, 0], X_test[:, 1], color='green', marker='^', label='test data')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend()




def run_app():
    app = QApplication(sys.argv)
    ui = KNNClassifierUI()
    ui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    run_app()
