import os

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QDoubleValidator, QIntValidator
from PyQt5.QtWidgets import (
    QComboBox,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from ui.gui.common.custom_checkbox import CustomCheckBox
from ui.gui.configs.main_window_config import COLOR_SPACES, FILE_FORMATS
from ui.gui.styles.main_window_style import *


def get_qt_shadow_effect():
    """
    Returns a shadow effect for a widget.

    :return: QGraphicsDropShadowEffect
    """
    shadow_effect = QGraphicsDropShadowEffect()
    shadow_effect.setBlurRadius(8)
    shadow_effect.setOffset(0, 4)
    shadow_effect.setColor(QColor(0, 0, 0, 80))
    return shadow_effect


def get_format_name_by_id(format_id):
    """
    Returns the name of the file format by its id.

    :param format_id: int

    :return: str
    """
    for f in FILE_FORMATS:
        if f["id"] == format_id:
            return f["name"].lower()
    return None


def get_color_space_name_by_id(color_space_id):
    """
    Returns the name of the color space by its id.

    :param color_space_id: int

    :return: str
    """
    for c in COLOR_SPACES:
        if c["id"] == color_space_id:
            return c["name"]
    return None


def get_positive_float_validator():
    """
    Returns a validator for positive float numbers.

    :return: QDoubleValidator
    """
    validator = QDoubleValidator(0.0001, 1e10, 4)
    validator.setNotation(QDoubleValidator.StandardNotation)
    return validator


def get_pattern_validator(pattern):
    """
    Returns a validator for a given pattern.

    :param pattern: str

    :return: QRegExpValidator
    """
    from PyQt5.QtCore import QRegExp
    from PyQt5.QtGui import QRegExpValidator

    regex = QRegExp(pattern)
    return QRegExpValidator(regex)


def create_input_field(placeholder="", validator=None):
    """
    Creates a QLineEdit widget with a placeholder text and a validator.

    :param placeholder: str
    :param validator: QValidator

    :return: QLineEdit
    """
    input_field = QLineEdit()
    input_field.setPlaceholderText(placeholder)
    if validator:
        input_field.setValidator(validator)
    return input_field


def create_qt_dropdown(label, items, default_index=0):
    """
    Creates a dropdown widget with a label and items.

    :param label: str
    :param items: list

    :return: QComboBox
    """
    down_arrow_icon_path = os.path.join(
        os.path.dirname(__file__), "../assets/images/vector.png"
    )

    label = QLabel(label)
    label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    label.setStyleSheet(common_label_style)

    dropdown = QComboBox()
    dropdown.setMinimumWidth(120)
    dropdown.setMaximumWidth(200)
    dropdown.addItems(items)
    dropdown.setCurrentIndex(default_index)
    dropdown.setStyleSheet(generate_dropdown_style(down_arrow_icon_path))

    return label, dropdown


def create_qt_label(text, style):
    """
    Creates a label widget with a given text and style.

    :param text: str
    :param style: str

    :return: QLabel
    """
    title = QLabel(text)
    title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    title.setStyleSheet(style)
    return title


def create_qt_checkbox_widget(value, event, label, default=False):
    """
    Creates a checkbox widget with a given value, event, and label.

    :param value: bool
    :param event: function
    :param label: str

    :return: CustomCheckBox
    """
    widget = QWidget()
    layout = QHBoxLayout(widget)

    checkbox = CustomCheckBox(value)
    checkbox.stateChanged.connect(event)
    checkbox.setChecked(default)
    layout.addWidget(checkbox)

    label = QLabel(label)
    label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    label.setStyleSheet(common_label_style)
    layout.addWidget(label)

    return widget


def create_qt_button(text, event, size=100):
    """
    Creates a button widget with a given text, event.

    :param text: str
    :param event: function

    :return: QPushButton
    """
    button = QPushButton(text)
    button.setFixedWidth(size)
    button.setCursor(Qt.PointingHandCursor)
    button.setStyleSheet(common_button_style)
    button.clicked.connect(event)

    return button


def create_qt_input(text, event):
    """
    Creates an input with a given text, event.

    :param text: str
    :param event: function

    :return: QLineEdit
    """
    input = QLineEdit()
    input.setFixedSize(82, 37)
    input.setPlaceholderText("Type")
    input.setStyleSheet(extension2process_style)
    input.textChanged.connect(event)

    return input
