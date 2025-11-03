from ui.gui.styles.variables import *

analyze_button_style = f"""
    QPushButton {{
        font-family: {FONT_FAMILY};
        font-size: {FONT_SIZE_LARGE};
        font-style: {FONT_STYLE_ITALIC};
        font-weight: {FONT_WEIGHT_NORMAL};
        line-height: {LINE_HEIGHT_LARGE};
        text-align: 'center';
        border: none;
        border-radius: {BORDER_RADIUS};
        padding: {PADDING_SMALL};
    }}
    QPushButton:enabled {{
        color: {COLOR_HIGHLIGHT};
        background-color: {COLOR_ACTIVE_BACKGROUND};
    }}
    QPushButton:disabled {{
        color: {COLOR_PRIMARY};
        background-color: {COLOR_DISABLED};
    }}
    QPushButton:hover {{
        background-color: {COLOR_HOVER};
    }}
    QPushButton:pressed {{
        background-color: {COLOR_PRESSED};
    }}
"""


def generate_phase_button_style(is_active):
    if is_active:
        return f"""
            font-family: {FONT_FAMILY};
            font-size: {FONT_SIZE_SMALL};
            font-weight: {FONT_WEIGHT_BOLD};
            line-height: {LINE_HEIGHT_LARGE};
            text-align: 'center';
            color: {COLOR_PRIMARY};
            border: none;
            border-bottom: {BORDER_BOTTOM};
        """
    else:
        return f"""
            font-family: {FONT_FAMILY};
            font-size: {FONT_SIZE_SMALL};
            font-weight: {FONT_WEIGHT_BOLD};
            line-height: {LINE_HEIGHT_LARGE};
            text-align: 'center';
            color: {COLOR_SECONDARY};
            border: none;
        """


def generate_dropdown_style(path):
    return f"""
        QComboBox {{
            font-family: {FONT_FAMILY};
            font-size: {FONT_SIZE_MEDIUM};
            font-weight: {FONT_WEIGHT_NORMAL};
            line-height: {LINE_HEIGHT_LARGE};
            color: {COLOR_PRIMARY};
            background-color: {COLOR_BACKGROUND};
            border: 1px solid {COLOR_DISABLED};
            padding: {PADDING_MEDIUM};
            border-radius: {BORDER_RADIUS_NONE};
        }}
        QComboBox::drop-down {{
            width: 24px;
            height: 24px;
            padding: 0px 6px;
            subcontrol-position: center right;
            border-left: none;
        }}
        QComboBox::down-arrow {{
            image: url({path});
            width: {ICON_WIDTH};
            height: {ICON_HEIGHT};
        }}
    """


waiting_label_style = f"""
    font-family: {FONT_FAMILY};
    font-size: {FONT_SIZE_SMALL};
    font-weight: {FONT_WEIGHT_NORMAL};
    line-height: {LINE_HEIGHT_SMALL};
    text-align: 'left';
    color: {COLOR_PRIMARY};
"""

percentage_label_style = f"""
    font-family: {FONT_FAMILY};
    font-size: {FONT_SIZE_SMALL};
    font-weight: {FONT_WEIGHT_NORMAL};
    line-height: {LINE_HEIGHT_SMALL};
    text-align: right;
    color: {COLOR_PRIMARY};
"""


def generate_progress_bar_style(isInProgress=False):
    backgroundColor = COLOR_INPROGRESS_BAR if isInProgress else COLOR_ACTIVE_BACKGROUND

    return f"""
        QProgressBar {{
            background-color: {COLOR_DISABLED};
            border-radius: {BORDER_RADIUS};
        }}
        QProgressBar::chunk {{
            background-color: {backgroundColor};
            border-radius: {BORDER_RADIUS};
        }}
    """


stack_widget_style = f"""
    QStackedWidget#content {{
        border: 2px solid {COLOR_DISABLED};
    }}
"""

header_style = f"""
    QFrame#header {{
        background-color: {COLOR_BACKGROUND};
        border: 2px solid {COLOR_DISABLED};
    }}
"""

app_title_style = f"""
    font-family: {FONT_FAMILY};
    font-size: 15px;
    font-weight: {FONT_WEIGHT_BOLD};
    color: {COLOR_PRIMARY};
"""

close_button_style = """
    QToolButton {
        border: none;
        background: transparent;
    }
"""

footer_style = f"""
    QFrame#footer {{
        background-color: {COLOR_BACKGROUND};
        border: 2px solid {COLOR_DISABLED};
    }}
"""

next_phase_button_style = f"""
    QPushButton {{
        font-family: {FONT_FAMILY};
        font-size: {FONT_SIZE_LARGE};
        font-style: {FONT_STYLE_ITALIC};
        font-weight: {FONT_WEIGHT_NORMAL};
        line-height: {LINE_HEIGHT_LARGE};
        text-align: 'center';
        color: {COLOR_HIGHLIGHT};
        background-color: {COLOR_ACTIVE_BACKGROUND};
        border: none;
        padding: {PADDING_SMALL};
        border-radius: {BORDER_RADIUS_NONE};
    }}
    QPushButton:hover {{
        background-color: {COLOR_HOVER_ACTIVE_BACKGROUND};
    }}
    QPushButton:pressed {{
        background-color: {COLOR_PRESSED_ACTIVE_BACKGROUND};
    }}
"""


common_title_label_style = f"""
    font-family: {FONT_FAMILY};
    font-size: {FONT_SIZE_MEDIUM};
    font-weight: {FONT_WEIGHT_BOLD};
    line-height: {LINE_HEIGHT_MEDIUM};
    letter-spacing: 0.02em;
    text-align: 'left';
    color: {COLOR_PRIMARY};
"""

common_button_style = f"""
    QPushButton {{
        font-family: {FONT_FAMILY};
        font-size: {FONT_SIZE_MEDIUM};
        font-style: {FONT_STYLE_ITALIC};
        font-weight: {FONT_WEIGHT_NORMAL};
        line-height: {LINE_HEIGHT_MEDIUM};
        text-align: 'left';
        color: {COLOR_PRIMARY};
        background-color: {COLOR_DISABLED};
        border: none;
        padding: {PADDING_SMALL};
        border-radius: {BORDER_RADIUS_NONE};
    }}
    QPushButton:hover {{
        background-color: {COLOR_HOVER};
    }}
    QPushButton:pressed {{
        background-color: {COLOR_PRESSED};
    }}
"""

stop_resume_button_style = f"""
    QPushButton {{
        background-color: {COLOR_BACKGROUND};
        border: none;
        border-radius: 4px 0 0 0;
    }}
"""

output_color_label_style = f"""
    font-family: {FONT_FAMILY};
    font-size: {FONT_SIZE_MEDIUM};
    font-weight: {FONT_WEIGHT_NORMAL};
    line-height: {LINE_HEIGHT_MEDIUM};
    letter-spacing: 0.02em;
    text-align: 'left';
    color: #000000;
"""

common_label_style = f"""
    font-family: {FONT_FAMILY};
    font-size: {FONT_SIZE_MEDIUM};
    font-weight: {FONT_WEIGHT_NORMAL};
    line-height: {LINE_HEIGHT_MEDIUM};
    letter-spacing: 0.02em;
    text-align: 'left';
    color: {COLOR_PRIMARY};
"""

extension2process_style = f"""
    QLineEdit {{
        font-family: {FONT_FAMILY};
        font-size: {FONT_SIZE_MEDIUM};
        font-style: {FONT_STYLE_ITALIC};
        font-weight: {FONT_WEIGHT_NORMAL};
        line-height: {LINE_HEIGHT_MEDIUM};
        text-align: 'left';
        color: {COLOR_PRIMARY};
        padding: {PADDING_SMALL};
        border: 1px solid {COLOR_DISABLED};
        border-radius: {BORDER_RADIUS_NONE};
        background-color: {COLOR_BACKGROUND};
    }}
"""
