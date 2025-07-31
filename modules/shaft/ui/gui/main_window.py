import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from locales.localization import _
from ui.gui.common.footer import Footer
from ui.gui.common.header import Header
from ui.gui.contexts.pages_context import PagesContext
from ui.gui.pages.analysis_page import AnalysisPage
from ui.gui.pages.development_page import DevelopmentPage
from ui.gui.styles.main_window_style import *
from ui.gui.styles.variables import *


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.output_path = None

        self.setWindowTitle("Custom GUI")
        self.setFixedSize(999, 800)  # Set fixed window size

        # Remove default window title bar
        self.setWindowFlags(Qt.FramelessWindowHint)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        central_widget.setLayout(main_layout)

        # Header
        self.header = Header(self)
        main_layout.addWidget(self.header)

        # Stacked Widget for content
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setObjectName("content")
        self.stacked_widget.setFixedSize(999, 640)
        self.stacked_widget.setStyleSheet(stack_widget_style)
        main_layout.addWidget(self.stacked_widget)

        # Footer
        self.footer = Footer(self)
        main_layout.addWidget(self.footer)

        # Pages
        self.create_pages()

        # Show the first page
        self.show_page(0)

        self.setStyleSheet(f"background-color: {COLOR_BACKGROUND};")

    def create_pages(self):
        # Pages context
        context = PagesContext()

        # Phase 1 (Analysis)
        analysis_page = AnalysisPage(self, context)
        analysis_page.update_progress.connect(self.update_progress)
        analysis_page.worker_started.connect(
            lambda: self.footer.show_next_phase_button(False)
        )
        analysis_page.worker_finished.connect(self.worker_finished)
        self.stacked_widget.addWidget(analysis_page)

        # Phase 2 (Development)
        development_page = DevelopmentPage(self, context)
        development_page.update_progress.connect(self.update_progress)
        development_page.worker_started.connect(
            lambda: self.footer.show_next_phase_button(False)
        )
        development_page.worker_finished.connect(self.worker_finished)
        self.stacked_widget.addWidget(development_page)

        self.stacked_widget.currentChanged.connect(self.on_page_changed)

    def show_page(self, index):
        self.stacked_widget.setCurrentIndex(index)
        self.header.update_button_styles(index)

    def on_page_changed(self, index):
        current_page = self.stacked_widget.widget(index)

        # Call the on_page_changed method if it exists for updating the page
        if callable(current_page.on_page_changed):
            current_page.on_page_changed()

    def update_progress(self, value):
        self.footer.set_progress(value, "In progress")

    def worker_finished(self, stopped=False):
        if stopped:
            self.footer.set_progress(0, "Waiting to launch process")
            return

        self.footer.set_progress(100, "Complete")

        if self.stacked_widget.currentIndex() == 0:
            self.footer.show_next_phase_button(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
