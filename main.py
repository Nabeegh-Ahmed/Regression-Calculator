import sys
import pandas as pd
import numpy as np
import matplotlib
from sklearn.linear_model import LinearRegression
from PyQt5.QtWidgets import \
    (QMainWindow, QWidget, QAction, QLabel, QPushButton, QApplication, QFileDialog, QGridLayout, QVBoxLayout, QLineEdit,
     QMessageBox)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pylab as plt

matplotlib.use('QT5Agg')


class Regression(FigureCanvasQTAgg):
    def __init__(self):
        self.fig, self.regression_plot = plt.subplots()
        super(Regression, self).__init__(self.fig)
        self.regression_generator = LinearRegression()
        self.data = None
        self.x_label = None
        self.y_label = None
        self.x_frame = None
        self.y_frame = None
        self.data_categories = None

    def draw_graph(self):
        if self.x_frame is None or self.y_frame is None:
            self.regression_plot.cla()
            self.regression_plot.scatter([0, 1, 2, 3, 4], [10, 1, 20, 3, 40])
            self.fig.canvas.draw_idle()
        else:
            self.regression_plot.cla()
            self.regression_plot.scatter(self.x_frame, self.y_frame)
            self.regression_plot.plot(self.x_frame, self.regression_generator.predict(self.x_frame), color='Red', linewidth=4)
            self.fig.canvas.draw_idle()

    def set_data(self, x_label, y_label, file_path):
        self.x_label = x_label
        self.y_label = y_label
        self.data = pd.read_csv(file_path)
        self.data_categories = open(file_path).readline().split(',')
        self.data_categories[-1] = self.data_categories[-1].split('\n')[0]
        if self.x_label in self.data_categories and self.y_label in self.data_categories:
            self.x_frame = pd.DataFrame(self.data, columns=[self.x_label])
            self.y_frame = pd.DataFrame(self.data, columns=[self.y_label])
            self.regression_generator.fit(self.x_frame, self.y_frame)

    def predict(self, input_value):
        if self.x_frame is not None and self.y_frame is not None:
            return round(self.regression_generator.predict(np.array(input_value).reshape(-1, 1))[0][0], 2)


class GUI(QMainWindow):
    def __init__(self):
        super().__init__()

        # Variables that will be used for regression plot
        self.file_path = None
        self.x_label = None
        self.y_label = None

        # Basic Setup for the window
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle("Regression Calculator")

        # The Menu Bar initiation
        self.menu = None
        self.menubar()

        # Setting up the calculator layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.calculator_layout = QVBoxLayout()
        self.central_widget.setLayout(self.calculator_layout)

        # GUI Elements for data processing
        self.data_processor_gui()

        # The graph controller and regression calculator backend
        self.regression = Regression()
        self.regression.draw_graph()
        self.calculator_layout.addWidget(self.regression)

        # GUI Elements for making predictions with the regression model
        self.prediction_gui()

        self.show()

    def menubar(self):
        # The Menu Bar
        self.menu = self.menuBar()
        # Buttons in Menu Bar
        file_menu = self.menu.addMenu("File")
        # Sub-options for File Menu
        new_action = QAction("New", self)
        file_menu.addAction(new_action)
        edit_menu = self.menu.addMenu("Edit")
        undo_action = QAction("Undo", self)
        redo_action = QAction("Redo", self)
        edit_menu.addAction(undo_action)
        edit_menu.addAction(redo_action)
        help_menu = self.menu.addMenu("help")
        git_help = QAction("Help on GitHub", self)
        email_help = QAction("Email Me", self)
        help_menu.addAction(git_help)
        help_menu.addAction(email_help)

    def data_processor_gui(self):
        processing_grid = QGridLayout()
        open_file = QPushButton("Open a CSV")
        open_file.resize(open_file.sizeHint())
        open_file.clicked.connect(self.open_file_name_dialog)
        processing_grid.addWidget(open_file, 0, 0, 1, 5)

        x_label = QLabel("X Label", self)
        processing_grid.addWidget(x_label, 1, 0)
        x_quantity = QLineEdit(self)
        processing_grid.addWidget(x_quantity, 1, 1)
        y_label = QLabel("Y Label", self)
        processing_grid.addWidget(y_label, 1, 2)
        y_quantity = QLineEdit(self)
        processing_grid.addWidget(y_quantity, 1, 3)
        calculate_regression_btn = QPushButton("Calculate")
        calculate_regression_btn.setStyleSheet(open("styles/button.css").read())
        processing_grid.addWidget(calculate_regression_btn, 1, 4)

        # Handle Calculate Click event
        calculate_regression_btn.clicked.connect(lambda: self.label_handler(x_quantity.text(), y_quantity.text()))

        self.calculator_layout.addLayout(processing_grid)

    def prediction_gui(self):
        prediction_grid = QGridLayout()
        prediction_label = QLabel("Get Prediction of a value: ")
        prediction_grid.addWidget(prediction_label, 0, 0)
        prediction_number = QLineEdit(self)
        prediction_grid.addWidget(prediction_number, 0, 1)
        prediction_btn = QPushButton("Predict")
        prediction_grid.addWidget(prediction_btn, 0, 3)

        prediction_btn.clicked.connect(lambda: self.prediction_event(prediction_number.text()))

        self.calculator_layout.addLayout(prediction_grid)

    # Event Handlers
    def label_handler(self, x_label, y_label):
        self.x_label = x_label
        self.y_label = y_label
        self.regression.set_data(self.x_label, self.y_label, self.file_path)
        self.regression.draw_graph()

    def open_file_name_dialog(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "CSV Files (*.csv)", options=options)
        if file_name and file_name.split('.')[-1] == "csv":
            self.file_path = file_name
            QMessageBox.about(self, "Loading a CSV", "CSV Loaded Successfully")
        else:
            QMessageBox.about(self, "Loading a CSV", "Loading Failed")

    def prediction_event(self, input_value):
        QMessageBox.about(self, "Prediction", str(self.regression.predict(int(input_value))))


def main():
    app = QApplication(sys.argv)
    ex = GUI()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


