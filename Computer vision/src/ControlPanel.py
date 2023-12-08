from PyQt5.QtWidgets import QApplication, QStyleFactory, QWidget, QVBoxLayout, QPushButton
from PyQt5.QtGui import QColor, QPalette


def create_and_show_control_panel():
    app = QApplication([])

    # Set Fusion style
    app.setStyle(QStyleFactory.create('Fusion'))

    # Modify color palette for a darker look
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    app.setPalette(palette)

    window = QWidget()
    layout = QVBoxLayout()

    # Add a button
    exit_button = QPushButton('Activate autopilot')
    exit_button.clicked.connect(app.exit)
    layout.addWidget(exit_button)

    window.setLayout(layout)
    window.show()
    app.exec_()
