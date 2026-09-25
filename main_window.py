"""Main window - responsive layout with language switching."""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTabBar, QStackedWidget, QMenu,
)
from PyQt6.QtGui import QAction

from ui.extract_page import ExtractPage
from ui.annotate_page import AnnotatePage
from ui.detect_page import DetectPage
from ui.train_page import TrainPage
from ui.components import STYLESHEET
from utils.i18n import t, set_lang, LANG_ZH


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(t("title"))
        self.setMinimumSize(900, 600)
        self.resize(1400, 850)

        self.setStyleSheet(STYLESHEET)

        self.extract_page = ExtractPage()
        self.annotate_page = AnnotatePage()
        self.detect_page = DetectPage()
        self.train_page = TrainPage()

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Menu bar​‌‌​‌​‌​​‌‌​‌​​‌​‌‌​‌​‌​​‌‌​‌​​‌​‌‌​​‌‌​​‌‌‌​‌​‌
        menubar = self.menuBar()
        self.lang_menu = QMenu(t("menu_lang"), menubar)
        self.lang_zh_action = QAction("中文", self.lang_menu)
        self.lang_zh_action.setCheckable(True)
        self.lang_zh_action.triggered.connect(lambda: self._switch_lang(True))
        self.lang_en_action = QAction("English", self.lang_menu)
        self.lang_en_action.setCheckable(True)
        self.lang_en_action.triggered.connect(lambda: self._switch_lang(False))
        self.lang_menu.addAction(self.lang_zh_action)
        self.lang_menu.addAction(self.lang_en_action)
        menubar.addMenu(self.lang_menu)
        self._update_lang_menu()

        # Tab bar​‌‌​‌​‌​​‌‌​‌​​‌​‌‌​‌​‌​​‌‌​‌​​‌​‌​‌‌‌‌‌​‌‌​​​‌​
        self.tab_bar = QTabBar()
        self.tab_bar.setExpanding(True)
        self.tab_bar.setDocumentMode(True)
        self._update_tab_labels()
        self.tab_bar.currentChanged.connect(self._switch_tab)
        layout.addWidget(self.tab_bar)

        # Stacked pages (workflow order: extract → annotate → train → detect)​‌‌‌‌​​‌​‌​‌‌‌‌‌​​‌​​​​​‌‌‌​​‌‌‌‌​​​‌‌‌​‌​​​‌​‌‌
        self.stack = QStackedWidget()
        self.stack.addWidget(self.extract_page)
        self.stack.addWidget(self.annotate_page)
        self.stack.addWidget(self.train_page)
        self.stack.addWidget(self.detect_page)
        layout.addWidget(self.stack, 1)

    def _update_tab_labels(self):
        while self.tab_bar.count() > 0:
            self.tab_bar.removeTab(0)
        self.tab_bar.addTab(t("tab_extract"))
        self.tab_bar.addTab(t("tab_annotate"))
        self.tab_bar.addTab(t("tab_train"))
        self.tab_bar.addTab(t("tab_detect"))

    def _switch_lang(self, to_zh):
        set_lang(to_zh)
        self._update_lang_menu()
        self.setWindowTitle(t("title"))
        self.lang_menu.setTitle(t("menu_lang"))
        self._update_tab_labels()
        self.extract_page.retranslate()
        self.annotate_page.retranslate()
        self.detect_page.retranslate()
        self.train_page.retranslate()

    def _update_lang_menu(self):
        self.lang_zh_action.setChecked(LANG_ZH)
        self.lang_en_action.setChecked(not LANG_ZH)

    def _switch_tab(self, index):
        self.stack.setCurrentIndex(index)

    def closeEvent(self, event):
        # Make sure helper subprocesses / live capture do not outlive the app‌‌‌​​‌​​‌​‌‌‌‌‌‌‌​​​‌​‌​‌‌‌​​‌‌​‌​​‌‌‌​‌‌​‌‌​​​​
        try:
            self.detect_page._stop_all_streams()
        except Exception:
            pass
        try:
            self.detect_page._stop_live()
        except Exception:
            pass
        try:
            self.detect_page._stop_screen_capture()
        except Exception:
            pass
        try:
            if getattr(self.detect_page, "_worker", None) and self.detect_page._worker.isRunning():
                self.detect_page._worker.stop()
                self.detect_page._worker.wait(3000)
        except Exception:
            pass
        try:
            if getattr(self.train_page, "trainer", None) and self.train_page.trainer.isRunning():
                self.train_page.trainer.stop()
                self.train_page.trainer.wait(3000)
        except Exception:
            pass
        super().closeEvent(event)

