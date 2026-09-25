"""Detection page - detect objects in images, videos, and camera."""

from __future__ import annotations

import os
import csv
import base64
from pathlib import Path
from datetime import datetime

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QFileDialog, QDoubleSpinBox, QSpinBox, QComboBox,
    QProgressBar, QMessageBox, QCheckBox, QGridLayout, QStackedWidget,
    QSplitter, QFrame,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QImage

from ui.components import styled_button
from utils.i18n import t


class DetectionSubprocess(QThread):
    """Run detection via system Python subprocess."""
    progress = pyqtSignal(int, int)
    frame_ready = pyqtSignal(object)
    detections_ready = pyqtSignal(list)
    log = pyqtSignal(str)
    done = pyqtSignal(int)

    def __init__(self, helper_script, model_path, source, output_dir, conf, iou, mode):
        super().__init__()
        self.helper_script = helper_script
        self.model_path = model_path
        self.source = source
        self.output_dir = output_dir
        self.conf = conf
        self.iou = iou
        self.mode = mode
        self._stop = False
        self._proc = None

    def stop(self):
        self._stop = True
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        import subprocess
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, timeout=5,
            )
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def run(self):
        import subprocess
        import json
        import tempfile
        import cv2

        cfg = {
            "action": self.mode,
            "model": self.model_path,
            "source": self.source,
            "output_dir": self.output_dir,
            "conf": self.conf,
            "iou": self.iou,
        }
        fd, cfg_path = tempfile.mkstemp(suffix=".json", prefix="_detect_cfg_")
        os.close(fd)
        with open(cfg_path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False)

        cmd = ["py", self.helper_script, cfg_path]

        try:
            self.log.emit("Starting detection...")
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=False, bufsize=0,
                startupinfo=startupinfo,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            proc = self._proc

            # Stream stdout (video can exceed any fixed timeout)
            stdout_data = []
            while not self._stop:
                line = proc.stdout.readline()
                if not line:
                    break
                stdout_data.append(line)
                text = line.decode('utf-8', errors='ignore').rstrip("\r\n")
                if not text:
                    continue
                if text.startswith('JSON:'):
                    try:
                        dets = json.loads(text[5:])
                        self.detections_ready.emit(dets)
                    except Exception:
                        pass
                elif text.startswith('FRAME:'):
                    parts = text.split(':', 3)
                    if len(parts) >= 4:
                        try:
                            frame_idx = int(parts[1])
                            total_frames = int(parts[2])
                            self.progress.emit(frame_idx + 1, total_frames)
                            frame_bytes = base64.b64decode(parts[3])
                            nparr = np.frombuffer(frame_bytes, np.uint8)
                            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if img is not None:
                                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                self.frame_ready.emit(rgb)
                        except Exception:
                            pass
                elif text.startswith('ERROR:') or text.startswith('DEBUG:'):
                    self.log.emit(text)

            if self._stop:
                self.stop()

            try:
                stderr = proc.stderr.read()
                if stderr:
                    err_text = stderr.decode('utf-8', errors='ignore')
                    if err_text.strip():
                        self.log.emit(f"Log: {err_text[:300]}")
                returncode = proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
                returncode = proc.returncode if proc.returncode is not None else -1

            if self._stop:
                self.done.emit(0)
                return

            if returncode != 0:
                self.log.emit(f"Process failed with code {returncode}")
                self.done.emit(0)
                return

            # Read output image for preview (from temp dir to avoid Chinese path issues)
            if self.mode == "image":
                temp_path = os.path.join(tempfile.gettempdir(), "yolo_detect_output", "_detected.jpg")
                if os.path.exists(temp_path):
                    img = cv2.imread(temp_path)
                    if img is not None:
                        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        self.frame_ready.emit(rgb)

            self.done.emit(1)
            self.log.emit("Done!")

        except Exception as e:
            self.log.emit(f"Error: {str(e)}")
            self.done.emit(0)
        finally:
            self._proc = None
            try:
                os.unlink(cfg_path)
            except OSError:
                pass



class StreamCaptureThread(QThread):
    """Capture one camera/RTSP source at a target FPS and emit frames."""

    frame_ready = pyqtSignal(str, object)  # stream_id, BGR ndarray
    status_changed = pyqtSignal(str, str)  # stream_id, message
    stopped = pyqtSignal(str)

    def __init__(self, stream_id, source, fps=10.0, target_size=None, parent=None):
        super().__init__(parent)
        self.stream_id = stream_id
        self.source = source  # int index or str URL
        self._fps = float(fps)
        self.target_size = target_size  # (w, h) or None
        self._stop = False

    def stop(self):
        self._stop = True

    def set_fps(self, fps):
        self._fps = max(1.0, float(fps))

    def set_target_size(self, size):
        self.target_size = size

    def run(self):
        import time
        import cv2

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.status_changed.emit(self.stream_id, t("cannot_open_rtsp")
                                     if isinstance(self.source, str)
                                     else t("cannot_open_camera"))
            self.stopped.emit(self.stream_id)
            return

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if isinstance(self.source, int):
            try:
                cap.set(cv2.CAP_PROP_FPS, self._fps)
            except Exception:
                pass

        self.status_changed.emit(self.stream_id, "OK")
        reconnect_delay = 0.5

        while not self._stop:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                if isinstance(self.source, str):
                    # RTSP drop → try reconnect
                    self.status_changed.emit(self.stream_id, "reconnect...")
                    cap.release()
                    time.sleep(reconnect_delay)
                    cap = cv2.VideoCapture(self.source)
                    if not cap.isOpened():
                        time.sleep(1.0)
                        continue
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass
                    continue
                self.status_changed.emit(self.stream_id, "read fail")
                break

            if self.target_size:
                tw, th = self.target_size
                if tw > 0 and th > 0 and (frame.shape[1], frame.shape[0]) != (tw, th):
                    frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)

            self.frame_ready.emit(self.stream_id, frame)

            interval = 1.0 / max(1.0, self._fps)
            elapsed = time.time() - t0
            sleep_s = interval - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

        try:
            cap.release()
        except Exception:
            pass
        self.status_changed.emit(self.stream_id, "stopped")
        self.stopped.emit(self.stream_id)


class DetectHub(QThread):
    """Single detect_server process shared by all live streams (queued)."""

    result_ready = pyqtSignal(str, object, list)  # sid, annotated BGR, dets
    log = pyqtSignal(str)

    def __init__(self, model_path, server_script, parent=None):
        super().__init__(parent)
        import queue
        self.model_path = model_path
        self.server_script = server_script
        self._queue = queue.Queue(maxsize=8)
        self._stop = False
        self._proc = None
        self._busy = False

    def submit(self, sid, frame, conf, iou):
        """Queue a frame; drop if the hub is full (keeps UI responsive)."""
        if self._stop:
            return
        item = (sid, frame, conf, iou)
        try:
            self._queue.put_nowait(item)
        except Exception:
            pass

    def stop(self):
        self._stop = True
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        proc = self._proc
        if proc is not None and proc.poll() is None:
            import subprocess
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True, timeout=5,
                )
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def run(self):
        import subprocess
        import json
        import base64
        import cv2

        try:
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            self._proc = subprocess.Popen(
                ["py", self.server_script],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=False, bufsize=0,
                startupinfo=si, creationflags=subprocess.CREATE_NO_WINDOW,
            )
            load = json.dumps({"action": "load", "model": self.model_path}) + "\n"
            self._proc.stdin.write(load.encode())
            self._proc.stdin.flush()
            resp = self._proc.stdout.readline()
            if not resp:
                err = self._proc.stderr.read() if self._proc.stderr else b""
                self.log.emit("Detect hub load failed: " + err.decode("utf-8", "ignore")[:200])
                return
            r = json.loads(resp.decode("utf-8", "ignore"))
            if r.get("status") != "ok":
                self.log.emit(f"Detect hub load error: {r}")
                return
            self.log.emit(t("server_ok"))

            while not self._stop:
                item = self._queue.get()
                if item is None:
                    break
                sid, frame, conf, iou = item
                # Drop backlog: if more than 2 waiting, process newest only
                while self._queue.qsize() > 2:
                    nxt = self._queue.get()
                    if nxt is None:
                        item = None
                        break
                    sid, frame, conf, iou = nxt
                if item is None:
                    break

                try:
                    self._busy = True
                    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    frame_b64 = base64.b64encode(buf).decode("utf-8")
                    cmd = json.dumps({
                        "action": "detect",
                        "id": sid,
                        "frame": frame_b64,
                        "conf": conf,
                        "iou": iou,
                    }) + "\n"
                    self._proc.stdin.write(cmd.encode())
                    self._proc.stdin.flush()
                    line = self._proc.stdout.readline()
                    if not line:
                        break
                    resp = json.loads(line.decode("utf-8", "ignore"))
                    out_sid = resp.get("id") or sid
                    if resp.get("status") == "ok" and resp.get("frame"):
                        fb = base64.b64decode(resp["frame"])
                        img = cv2.imdecode(np.frombuffer(fb, np.uint8), cv2.IMREAD_COLOR)
                        if img is not None:
                            self.result_ready.emit(out_sid, img, resp.get("detections") or [])
                    elif resp.get("status") != "ok":
                        self.log.emit(f"{out_sid}: {resp.get('msg', 'error')}")
                except Exception as e:
                    self.log.emit(f"Detect hub: {e}")
                    break
                finally:
                    self._busy = False
        except Exception as e:
            self.log.emit(f"Detect hub crashed: {e}")
        finally:
            if self._proc is not None:
                try:
                    if self._proc.poll() is None:
                        self._proc.stdin.write(b'{"action":"quit"}\n')
                        self._proc.stdin.flush()
                        self._proc.wait(timeout=2)
                except Exception:
                    try:
                        self._proc.kill()
                    except Exception:
                        pass
                self._proc = None


def build_rtsp_url(base_url, user="", password=""):
    """Compose rtsp://user:pass@host/path from separate fields."""
    from urllib.parse import urlparse, urlunparse, quote
    base_url = (base_url or "").strip()
    if not base_url:
        return ""
    if "://" not in base_url:
        base_url = "rtsp://" + base_url
    parsed = urlparse(base_url)
    if parsed.scheme not in ("rtsp", "rtsps"):
        # allow odd schemes to pass through for OpenCV
        pass
    netloc = parsed.netloc
    if "@" in netloc:
        netloc = netloc.split("@", 1)[-1]
    auth = ""
    if user or password:
        auth = quote(user or "", safe="")
        if password:
            auth += ":" + quote(password, safe="")
        auth += "@"
    return urlunparse((parsed.scheme or "rtsp", auth + netloc,
                       parsed.path, parsed.params, parsed.query, parsed.fragment))



class StreamWindow(QWidget):
    """Floating window for one camera/RTSP stream."""

    request_close = pyqtSignal(str)

    def __init__(self, sid, title, parent=None):
        super().__init__(parent)
        self.sid = sid
        self.setWindowTitle(title)
        self.setMinimumSize(420, 320)
        self.resize(640, 480)
        self.setStyleSheet("""
            QWidget { background: #111827; color: #e5e7eb; }
            QLabel#hint { color: #6b7280; font-size: 12px; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.preview = QLabel("...")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(320, 240)
        self.preview.setStyleSheet("""
            QLabel { background: #0c1017; border: 2px dashed #374151;
            border-radius: 8px; color: #6b7280; font-size: 14px; }
        """)
        layout.addWidget(self.preview, 1)

        bar = QHBoxLayout()
        bar.setSpacing(8)
        self.detect_chk = QCheckBox(t("detect_on"))
        bar.addWidget(self.detect_chk)
        bar.addWidget(QLabel(t("fps")))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setMaximumWidth(60)
        bar.addWidget(self.fps_spin)
        self.status = QLabel("...")
        self.status.setObjectName("hint")
        bar.addWidget(self.status, 1)
        self.close_btn = styled_button(t("remove_stream"), "#ef4444", 11)
        self.close_btn.setMaximumWidth(80)
        self.close_btn.clicked.connect(self._on_close_clicked)
        bar.addWidget(self.close_btn)
        layout.addLayout(bar)

    def _on_close_clicked(self):
        if not getattr(self, "_closing", False):
            self._closing = True
            self.request_close.emit(self.sid)
        self.close()

    def closeEvent(self, event):
        if not getattr(self, "_closing", False):
            self._closing = True
            self.request_close.emit(self.sid)
        super().closeEvent(event)


class DetectPage(QWidget):
    """Detection page for images, videos, and camera."""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._model_path = None
        self._worker = None
        self._camera_cap = None
        self._detect_proc = None
        # legacy single-source timer removed — multi-stream uses StreamCaptureThread
        self._all_detections = []
        self._screen_capture = None
        self._screen_region = None  # (x, y, w, h) or None
        self._active_source = None  # "cam" | "rtsp" (legacy)
        self._streams = {}  # sid -> dict(thread, detect, label, source, kind, fps, res, row)
        self._stream_order = []
        self._next_sid = 1
        self._hub = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)

        # ── Left: preview ──
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(8)

        self.preview_label = QLabel(t("no_model_loaded"))
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(320, 240)
        self.preview_label.setStyleSheet("""
            QLabel { background: #0c1017; border: 2px dashed #374151;
            border-radius: 12px; color: #6b7280; font-size: 16px; }
        """)
        left_layout.addWidget(self.preview_label, 1)

        left_bar = QHBoxLayout()
        left_bar.setSpacing(8)
        self.status_label = QLabel(t("ready"))
        self.status_label.setObjectName("hint")
        left_bar.addWidget(self.status_label, 1)
        self.stream_status_label = QLabel(t("no_streams"))
        self.stream_status_label.setObjectName("hint")
        left_bar.addWidget(self.stream_status_label)
        left_layout.addLayout(left_bar)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        left_layout.addWidget(self.progress_bar)

        splitter.addWidget(left)

        # ── Right: controls (like annotate page) ──
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(10)

        # Model​‌‌​‌​‌​​‌‌​‌​​‌​‌‌​‌​‌​​‌‌​‌​​‌​‌‌​​‌‌​​‌‌‌​‌​‌
        model_sec = QLabel(t("select_model_pt"))
        model_sec.setObjectName("sectionTitle")
        right_layout.addWidget(model_sec)

        model_row = QHBoxLayout()
        model_row.setSpacing(8)
        self.model_section_label = model_sec
        self.model_label = QLabel(t("no_model_loaded"))
        self.model_label.setObjectName("hint")
        self.model_label.setWordWrap(True)
        model_row.addWidget(self.model_label, 1)
        self.load_btn = styled_button(t("load_model"), "#f97316", 12)
        self.load_btn.setMaximumWidth(110)
        self.load_btn.clicked.connect(self._load_model)
        model_row.addWidget(self.load_btn)
        right_layout.addLayout(model_row)

        div1 = QFrame()
        div1.setFrameShape(QFrame.Shape.HLine)
        div1.setStyleSheet("color: #374151;")
        right_layout.addWidget(div1)

        # Thresholds
        thr_sec = QLabel(t("params"))
        thr_sec.setObjectName("sectionTitle")
        right_layout.addWidget(thr_sec)

        thr1 = QHBoxLayout()
        thr1.setSpacing(8)
        self.conf_label = QLabel(t("confidence"))
        self.conf_label.setObjectName("sectionTitleSmall")
        thr1.addWidget(self.conf_label)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setValue(0.25)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setMaximumWidth(90)
        thr1.addWidget(self.conf_spin)
        self.iou_label = QLabel(t("iou_thresh"))
        self.iou_label.setObjectName("sectionTitleSmall")
        thr1.addWidget(self.iou_label)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.1, 1.0)
        self.iou_spin.setValue(0.45)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setDecimals(2)
        self.iou_spin.setMaximumWidth(90)
        thr1.addWidget(self.iou_spin)
        thr1.addStretch(1)
        right_layout.addLayout(thr1)

        div2 = QFrame()
        div2.setFrameShape(QFrame.Shape.HLine)
        div2.setStyleSheet("color: #374151;")
        right_layout.addWidget(div2)

        # Detect sources
        src_sec = QLabel(t("tab_detect"))
        src_sec.setObjectName("sectionTitle")
        right_layout.addWidget(src_sec)

        def vbtn(text_key, color, slot):
            b = styled_button(t(text_key), color, 12)
            b.clicked.connect(slot)
            right_layout.addWidget(b)
            return b

        self.img_btn = vbtn("detect_image", "#3b82f6", self._detect_image)
        self.video_btn = vbtn("detect_video", "#8b5cf6", self._detect_video)
        self.rtsp_btn = vbtn("rtsp", "#06b6d4", self._toggle_rtsp_settings)
        self.cam_btn = vbtn("detect_camera", "#22c55e", self._toggle_camera_settings)
        self.screen_btn = vbtn("screen_capture", "#eab308", self._toggle_screen_settings)

        # Camera settings (vertical form, right panel)
        self.cam_settings = QWidget()
        cam_form = QVBoxLayout(self.cam_settings)
        cam_form.setContentsMargins(0, 4, 0, 0)
        cam_form.setSpacing(6)

        r_cam = QHBoxLayout()
        self.camera_label = QLabel(t("camera"))
        self.camera_label.setObjectName("sectionTitleSmall")
        r_cam.addWidget(self.camera_label)
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(120)
        r_cam.addWidget(self.camera_combo, 1)
        self.refresh_cam_btn = styled_button(t("refresh_cameras"), "#6b7280", 11)
        self.refresh_cam_btn.setMaximumWidth(56)
        self.refresh_cam_btn.clicked.connect(self._refresh_cameras)
        r_cam.addWidget(self.refresh_cam_btn)
        cam_form.addLayout(r_cam)
        self._refresh_cameras()

        r_cam2 = QHBoxLayout()
        self.cam_res_label = QLabel(t("resolution"))
        self.cam_res_label.setObjectName("sectionTitleSmall")
        r_cam2.addWidget(self.cam_res_label)
        self.cam_res_combo = QComboBox()
        self.cam_res_combo.addItems(["640x480", "800x600", "1280x720", "1920x1080", "原始"])
        self.cam_res_combo.setCurrentText("640x480")
        r_cam2.addWidget(self.cam_res_combo, 1)
        self.cam_fps_label = QLabel(t("fps"))
        self.cam_fps_label.setObjectName("sectionTitleSmall")
        r_cam2.addWidget(self.cam_fps_label)
        self.cam_fps_spin = QSpinBox()
        self.cam_fps_spin.setRange(1, 60)
        self.cam_fps_spin.setValue(10)
        self.cam_fps_spin.setMaximumWidth(56)
        r_cam2.addWidget(self.cam_fps_spin)
        cam_form.addLayout(r_cam2)

        self.cam_start_btn = styled_button(t("add_stream"), "#22c55e", 12)
        self.cam_start_btn.clicked.connect(self._add_camera_stream)
        cam_form.addWidget(self.cam_start_btn)
        self.cam_settings.setVisible(False)
        right_layout.addWidget(self.cam_settings)

        # RTSP settings
        self.rtsp_settings = QWidget()
        rtsp_form = QVBoxLayout(self.rtsp_settings)
        rtsp_form.setContentsMargins(0, 4, 0, 0)
        rtsp_form.setSpacing(6)

        self.rtsp_url_edit = QLineEdit()
        self.rtsp_url_edit.setPlaceholderText(t("rtsp_hint"))
        self.rtsp_url_edit.returnPressed.connect(self._rtsp_confirm)
        rtsp_form.addWidget(self.rtsp_url_edit)

        r_u = QHBoxLayout()
        self.rtsp_user_label = QLabel(t("rtsp_user"))
        self.rtsp_user_label.setObjectName("sectionTitleSmall")
        r_u.addWidget(self.rtsp_user_label)
        self.rtsp_user_edit = QLineEdit()
        r_u.addWidget(self.rtsp_user_edit, 1)
        rtsp_form.addLayout(r_u)

        r_p = QHBoxLayout()
        self.rtsp_pass_label = QLabel(t("rtsp_pass"))
        self.rtsp_pass_label.setObjectName("sectionTitleSmall")
        r_p.addWidget(self.rtsp_pass_label)
        self.rtsp_pass_edit = QLineEdit()
        self.rtsp_pass_edit.setEchoMode(QLineEdit.EchoMode.Password)
        r_p.addWidget(self.rtsp_pass_edit, 1)
        rtsp_form.addLayout(r_p)

        r_rtsp2 = QHBoxLayout()
        self.rtsp_res_label = QLabel(t("resolution"))
        self.rtsp_res_label.setObjectName("sectionTitleSmall")
        r_rtsp2.addWidget(self.rtsp_res_label)
        self.rtsp_res_combo = QComboBox()
        self.rtsp_res_combo.addItems(["640x480", "800x600", "1280x720", "1920x1080", "原始"])
        self.rtsp_res_combo.setCurrentText("640x480")
        r_rtsp2.addWidget(self.rtsp_res_combo, 1)
        self.rtsp_fps_label = QLabel(t("fps"))
        self.rtsp_fps_label.setObjectName("sectionTitleSmall")
        r_rtsp2.addWidget(self.rtsp_fps_label)
        self.rtsp_fps_spin = QSpinBox()
        self.rtsp_fps_spin.setRange(1, 60)
        self.rtsp_fps_spin.setValue(10)
        self.rtsp_fps_spin.setMaximumWidth(56)
        r_rtsp2.addWidget(self.rtsp_fps_spin)
        rtsp_form.addLayout(r_rtsp2)

        self.rtsp_hint_label = QLabel(t("rtsp_auth_hint"))
        self.rtsp_hint_label.setObjectName("hint")
        self.rtsp_hint_label.setWordWrap(True)
        rtsp_form.addWidget(self.rtsp_hint_label)

        self.rtsp_confirm_btn = styled_button(t("add_stream"), "#06b6d4", 12)
        self.rtsp_confirm_btn.clicked.connect(self._rtsp_confirm)
        rtsp_form.addWidget(self.rtsp_confirm_btn)
        self.rtsp_settings.setVisible(False)
        right_layout.addWidget(self.rtsp_settings)

        # Screen capture​‌‌​‌​‌​​‌‌​‌​​‌​‌‌​‌​‌​​‌‌​‌​​‌​‌​‌‌‌‌‌​‌‌​​​‌​
        self.screen_settings = QWidget()
        scr_form = QVBoxLayout(self.screen_settings)
        scr_form.setContentsMargins(0, 4, 0, 0)
        scr_form.setSpacing(6)

        r_mon = QHBoxLayout()
        self.monitor_label = QLabel(t("monitor"))
        self.monitor_label.setObjectName("sectionTitleSmall")
        r_mon.addWidget(self.monitor_label)
        self.monitor_combo = QComboBox()
        r_mon.addWidget(self.monitor_combo, 1)
        scr_form.addLayout(r_mon)

        r_scr2 = QHBoxLayout()
        self.fps_label = QLabel(t("fps"))
        self.fps_label.setObjectName("sectionTitleSmall")
        r_scr2.addWidget(self.fps_label)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 30)
        self.fps_spin.setValue(5)
        self.fps_spin.setMaximumWidth(56)
        r_scr2.addWidget(self.fps_spin)
        self.region_btn = styled_button(t("select_region"), "#6b7280", 11)
        self.region_btn.setMaximumWidth(100)
        self.region_btn.clicked.connect(self._select_region)
        r_scr2.addWidget(self.region_btn)
        self.region_label = QLabel(t("full_screen"))
        self.region_label.setObjectName("hint")
        r_scr2.addWidget(self.region_label, 1)
        scr_form.addLayout(r_scr2)

        self.start_screen_btn = styled_button(t("start"), "#22c55e", 12)
        self.start_screen_btn.clicked.connect(self._start_screen_capture)
        scr_form.addWidget(self.start_screen_btn)
        self.screen_settings.setVisible(False)
        right_layout.addWidget(self.screen_settings)

        div3 = QFrame()
        div3.setFrameShape(QFrame.Shape.HLine)
        div3.setStyleSheet("color: #374151;")
        right_layout.addWidget(div3)

        # Export​‌‌‌‌​​‌​‌​‌‌‌‌‌​​‌​​​​​‌‌‌​​‌‌‌‌​​​‌‌‌​‌​​​‌​‌‌
        exp_sec = QLabel(t("export_result"))
        exp_sec.setObjectName("sectionTitle")
        right_layout.addWidget(exp_sec)
        exp_row = QHBoxLayout()
        exp_row.setSpacing(8)
        self.csv_btn = styled_button(t("export_csv"), "#6b7280", 11)
        self.csv_btn.clicked.connect(self._export_csv)
        exp_row.addWidget(self.csv_btn)
        self.save_img_btn = styled_button(t("save_annotated"), "#6b7280", 11)
        self.save_img_btn.clicked.connect(self._save_last_annotated)
        exp_row.addWidget(self.save_img_btn)
        right_layout.addLayout(exp_row)

        # Hidden legacy labels some code still sets
        self.stream_list_label = QLabel(t("stream_list"))
        self.stream_list_label.setVisible(False)
        right_layout.addWidget(self.stream_list_label)

        right_layout.addStretch(1)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([920, 320])
        layout.addWidget(splitter, 1)

    def retranslate(self):
        self.model_section_label.setText(t("select_model_pt"))
        self.load_btn.setText(t("load_model"))
        self.img_btn.setText(t("detect_image"))
        self.video_btn.setText(t("detect_video"))
        self.rtsp_btn.setText(t("rtsp"))
        self.cam_btn.setText(t("detect_camera"))
        self.camera_label.setText(t("camera"))
        self.refresh_cam_btn.setText(t("refresh_cameras"))
        if self.camera_combo.count() == 1 and self.camera_combo.itemData(0) == -1:
            self.camera_combo.setItemText(0, t("no_camera"))
        self.cam_res_label.setText(t("resolution"))
        self.cam_fps_label.setText(t("fps"))
        self.rtsp_res_label.setText(t("resolution"))
        self.rtsp_fps_label.setText(t("fps"))
        self.rtsp_confirm_btn.setText(t("add_stream"))
        self.rtsp_url_edit.setPlaceholderText(t("rtsp_hint"))
        if hasattr(self, "rtsp_user_edit"):
            self.rtsp_user_label.setText(t("rtsp_user"))
            self.rtsp_pass_label.setText(t("rtsp_pass"))
            self.rtsp_user_edit.setPlaceholderText(t("rtsp_user"))
            self.rtsp_pass_edit.setPlaceholderText(t("rtsp_pass"))
            self.rtsp_hint_label.setText(t("rtsp_auth_hint"))
        self.cam_start_btn.setText(t("add_stream"))
        self._update_stream_status_label()
        self.csv_btn.setText(t("export_csv"))
        self.save_img_btn.setText(t("save_annotated"))
        self.conf_label.setText(t("confidence"))
        self.iou_label.setText(t("iou_thresh"))
        if not (self._screen_capture and self._screen_capture.isRunning()):
            self.screen_btn.setText(t("screen_capture"))
        self.monitor_label.setText(t("monitor"))
        self.fps_label.setText(t("fps"))
        self.region_btn.setText(t("select_region"))
        self.start_screen_btn.setText(t("start"))
        self.region_label.setText(t("full_screen"))
        if self.model_label.text() in ("未加载模型", "No model loaded"):
            self.model_label.setText(t("no_model_loaded"))
        if self.status_label.text() in ("就绪", "Ready"):
            self.status_label.setText(t("ready"))

    # ─── Model ───

    def _load_model(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, t("select_model_pt"), "", "YOLO Model (*.pt);;All (*)"
        )
        if not filepath:
            return
        self._model_path = filepath
        self.model_label.setText(Path(filepath).name)
        self.model_label.setStyleSheet("color: #22c55e; font-size: 12px;")
        # Enable per-stream detect toggles and start hub if any already want detect
        for info in self._streams.values():
            chk = info.get("detect_chk")
            if chk:
                chk.setEnabled(True)
                if not chk.isChecked():
                    chk.setChecked(True)
        self._ensure_hub_if_needed()

    # ─── Detect Image ───

    def _detect_image(self):
        if not self._model_path:
            QMessageBox.warning(self, t("title"), t("no_model_loaded"))
            return
        filepath, _ = QFileDialog.getOpenFileName(
            self, t("detect_image"), "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp);;All (*)"
        )
        if not filepath:
            return
        self.status_label.setText(t("detecting"))
        self.status_label.setStyleSheet("color: #f97316; font-size: 12px;")
        self._source_path = filepath
        # Save to original image's parent directory
        output_dir = str(Path(filepath).parent / "detection_results")
        helper = str(Path(__file__).parent.parent / "detect_helper.py")
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
        self._worker = DetectionSubprocess(
            helper, self._model_path, filepath, output_dir,
            self.conf_spin.value(), self.iou_spin.value(), "image"
        )
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.detections_ready.connect(self._on_dets)
        self._worker.log.connect(self._log)
        self._worker.done.connect(self._on_done)
        self._worker.start()

    # ─── Detect Video ───

    def _detect_video(self):
        if not self._model_path:
            QMessageBox.warning(self, t("title"), t("no_model_loaded"))
            return
        filepath, _ = QFileDialog.getOpenFileName(
            self, t("detect_video"), "",
            "Video (*.mp4 *.avi *.mov *.mkv *.wmv);;All (*)"
        )
        if not filepath:
            return
        self._detect_video_from_path(filepath)

    def _detect_video_from_path(self, filepath):
        if not self._model_path:
            QMessageBox.warning(self, t("title"), t("no_model_loaded"))
            return
        self._source_path = filepath
        # Save to video's parent directory‌‌‌​​‌​​‌​‌‌‌‌‌‌‌​​​‌​‌​‌‌‌​​‌‌​‌​​‌‌‌​‌‌​‌‌​​​​
        output_dir = str(Path(filepath).parent / "detection_results")
        self.status_label.setText(t("detecting"))
        self.status_label.setStyleSheet("color: #f97316; font-size: 12px;")
        helper = str(Path(__file__).parent.parent / "detect_helper.py")
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
        self._worker = DetectionSubprocess(
            helper, self._model_path, filepath, output_dir,
            self.conf_spin.value(), self.iou_spin.value(), "video"
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.detections_ready.connect(self._on_dets)
        self._worker.log.connect(self._log)
        self._worker.done.connect(self._on_done)
        self._worker.start()

    # ─── Multi-stream (camera / RTSP) ───

    def _parse_res_text(self, text_val):
        if not text_val or text_val == "原始" or text_val.lower() == "raw":
            return None
        try:
            w, h = text_val.split("x", 1)
            return int(w), int(h)
        except Exception:
            return None

    def _toggle_camera_settings(self):
        vis = not self.cam_settings.isVisible()
        if vis:
            self.rtsp_settings.setVisible(False)
        self.cam_settings.setVisible(vis)
        self._active_source = "cam" if vis else None

    def _toggle_rtsp_settings(self):
        vis = not self.rtsp_settings.isVisible()
        if vis:
            self.cam_settings.setVisible(False)
        self.rtsp_settings.setVisible(vis)
        self._active_source = "rtsp" if vis else None

    def _add_camera_stream(self):
        idx = self.camera_combo.currentData()
        if idx is None or idx < 0:
            QMessageBox.warning(self, t("title"), t("no_camera"))
            return
        sid_key = f"cam:{idx}"
        if sid_key in {s.get("key") for s in self._streams.values()}:
            self.status_label.setText(t("stream_dup"))
            return
        name = self.camera_combo.currentText() or f"Camera {idx}"
        self._start_stream(
            key=sid_key,
            name=name,
            source=idx,
            kind="cam",
            fps=self.cam_fps_spin.value(),
            res_text=self.cam_res_combo.currentText(),
            default_detect=True,
        )

    def _rtsp_confirm(self):
        raw = self.rtsp_url_edit.text().strip()
        if not raw:
            return
        user = self.rtsp_user_edit.text().strip()
        password = self.rtsp_pass_edit.text()
        url = build_rtsp_url(raw, user, password)
        # Dedup by host+path (ignore credentials for key)
        from urllib.parse import urlparse
        parsed = urlparse(url)
        sid_key = f"rtsp://{parsed.hostname or ''}{parsed.path}"
        if sid_key in {s.get("key") for s in self._streams.values()}:
            self.status_label.setText(t("stream_dup"))
            return
        name = parsed.hostname or url[:40]
        self._start_stream(
            key=sid_key,
            name=name,
            source=url,
            kind="rtsp",
            fps=self.rtsp_fps_spin.value(),
            res_text=self.rtsp_res_combo.currentText(),
            default_detect=True,
        )

    def _start_stream(self, key, name, source, kind, fps, res_text, default_detect=True):
        sid = f"s{self._next_sid}"
        self._next_sid += 1
        target = self._parse_res_text(res_text)

        thread = StreamCaptureThread(sid, source, fps=fps, target_size=target)
        thread.frame_ready.connect(self._on_stream_frame)
        thread.status_changed.connect(self._on_stream_status)
        thread.stopped.connect(self._on_stream_stopped)

        win = StreamWindow(sid, f"{name} [{kind}]")
        win.detect_chk.setChecked(bool(default_detect and self._model_path))
        win.detect_chk.setEnabled(bool(self._model_path))
        win.detect_chk.toggled.connect(lambda _=False: self._ensure_hub_if_needed())
        win.fps_spin.setValue(int(fps))
        win.fps_spin.valueChanged.connect(lambda v, s=sid: self._on_stream_fps(s, v))
        win.request_close.connect(self._remove_stream)
        win.destroyed.connect(lambda _=False, s=sid: self._on_stream_window_gone(s))

        self._streams[sid] = {
            "key": key,
            "name": name,
            "source": source,
            "kind": kind,
            "fps": int(fps),
            "res": target,
            "thread": thread,
            "detect_chk": win.detect_chk,
            "fps_spin": win.fps_spin,
            "status": win.status,
            "win": win,
            "closing": False,
        }
        self._stream_order.append(sid)

        # Cascade windows so they don't stack exactly
        n = len(self._stream_order)
        win.move(80 + (n - 1) * 36, 80 + (n - 1) * 28)
        win.show()
        win.raise_()
        win.activateWindow()

        thread.start()
        self.status_label.setText(t("stream_added", name=name))
        self._update_stream_status_label()
        self._ensure_hub_if_needed()

    def _update_stream_status_label(self):
        n = len(self._streams)
        if n == 0:
            self.stream_status_label.setText(t("no_streams"))
        else:
            names = ", ".join(self._streams[s]["name"] for s in self._stream_order if s in self._streams)
            self.stream_status_label.setText(t("open_windows", n=n, names=names))

    def _on_stream_window_gone(self, sid):
        # Window destroyed without going through _remove_stream cleanly
        if sid in self._streams and not self._streams[sid].get("closing"):
            self._remove_stream(sid, from_window=True)

    def _on_stream_fps(self, sid, fps):
        info = self._streams.get(sid)
        if not info:
            return
        info["fps"] = int(fps)
        th = info.get("thread")
        if th:
            th.set_fps(fps)
        spin = info.get("fps_spin")
        if spin and spin.value() != int(fps):
            spin.blockSignals(True)
            spin.setValue(int(fps))
            spin.blockSignals(False)

    def _on_stream_status(self, sid, msg):
        info = self._streams.get(sid)
        if info and info.get("status"):
            info["status"].setText(msg)

    def _on_stream_stopped(self, sid):
        self._on_stream_status(sid, "stopped")

    def _on_stream_frame(self, sid, frame):
        import cv2
        info = self._streams.get(sid)
        if not info:
            return
        show = frame
        conf = self.conf_spin.value()
        iou = self.iou_spin.value()
        detect = info["detect_chk"].isChecked() and self._hub is not None
        if detect:
            self._hub.submit(sid, frame, conf, iou)

        win = info.get("win")
        if win is not None:
            self._paint_tile(win.preview, show)
        rgb = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        self._last_frame = rgb

    def _on_hub_result(self, sid, annotated_bgr, dets):
        import cv2
        if dets:
            tagged = [dict(d, stream=sid) for d in dets]
            self._all_detections.extend(tagged)
            self._last_dets = dets
        info = self._streams.get(sid)
        if info:
            win = info.get("win")
            if win is not None:
                self._paint_tile(win.preview, annotated_bgr)
        rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        self._last_frame = rgb

    def _paint_tile(self, tile, frame):
        import cv2
        import numpy as np
        if frame is None or tile is None:
            return
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif len(frame.shape) == 3 and frame.shape[2] == 4:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            rgb = frame
        h, w, ch = rgb.shape
        data = np.ascontiguousarray(rgb)
        qimg = QImage(data.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())
        scaled = pixmap.scaled(
            tile.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        tile.setText("")
        tile.setPixmap(scaled)

    def _remove_stream(self, sid, from_window=False):
        info = self._streams.pop(sid, None)
        if not info:
            return
        info["closing"] = True
        th = info.get("thread")
        if th:
            th.stop()
            th.wait(2000)
        win = info.get("win")
        if win is not None:
            try:
                win.request_close.disconnect()
            except Exception:
                pass
            try:
                win.hide()
                win.deleteLater()
            except Exception:
                pass
        if sid in self._stream_order:
            self._stream_order.remove(sid)
        self._update_stream_status_label()
        if not self._streams:
            if self._hub:
                self._hub.stop()
                self._hub.wait(3000)
                self._hub = None
            if not (self._screen_capture and self._screen_capture.isRunning()):
                self._stop_detect_proc_legacy()

    def _stop_all_streams(self):
        for sid in list(self._streams.keys()):
            self._remove_stream(sid)

    def _ensure_hub_if_needed(self):
        need = any(
            info["detect_chk"].isChecked()
            for info in self._streams.values()
        )
        if need and self._hub is None and self._model_path:
            self._start_hub()
        if self._hub and not need and not (self._screen_capture and self._screen_capture.isRunning()):
            # keep hub if screen capture still needs it
            pass

    def _start_hub(self):
        if self._hub is not None:
            return
        if not self._model_path:
            self.status_label.setText(t("need_model_first"))
            return
        server = str(Path(__file__).parent.parent / "detect_server.py")
        self._hub = DetectHub(self._model_path, server)
        self._hub.result_ready.connect(self._on_hub_result)
        self._hub.log.connect(self._log)
        self._hub.start()

    def _stop_detect_proc_legacy(self):
        if self._detect_proc:
            try:
                self._detect_proc.stdin.write(b'{"action":"quit"}\n')
                self._detect_proc.stdin.flush()
                self._detect_proc.wait(timeout=3)
            except Exception:
                try:
                    self._detect_proc.kill()
                except Exception:
                    pass
            self._detect_proc = None

    def _stop_live(self):
        """Backward-compatible stop for old single-stream API + hub/streams."""
        self._stop_all_streams()
        if self._camera_cap:
            try:
                self._camera_cap.release()
            except Exception:
                pass
            self._camera_cap = None
        if self._hub:
            self._hub.stop()
            self._hub.wait(3000)
            self._hub = None
        self._stop_detect_proc_legacy()
        if hasattr(self, "cam_start_btn"):
            self.cam_start_btn.setText(t("add_stream"))
        if hasattr(self, "rtsp_confirm_btn"):
            self.rtsp_confirm_btn.setText(t("add_stream"))

    def _start_detect_server(self):
        """Start hub (preferred) or legacy single detect_proc for screen capture."""
        self._start_hub()
        if self._hub is None and not self._detect_proc:
            import subprocess
            import json
            server = str(Path(__file__).parent.parent / "detect_server.py")
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            self._detect_proc = subprocess.Popen(
                ["py", server], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=False,
                startupinfo=si, creationflags=subprocess.CREATE_NO_WINDOW,
            )
            cmd = json.dumps({"action": "load", "model": self._model_path}) + "\n"
            self._detect_proc.stdin.write(cmd.encode())
            self._detect_proc.stdin.flush()
            resp = self._detect_proc.stdout.readline()
            if resp:
                r = json.loads(resp.decode())
                self._log(f"{t('server_ok')}: {r.get('status', 'error')}")

    # ─── Camera list ───

    def _refresh_cameras(self):
        import cv2
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        names = self._camera_names()
        indexes = []
        for i in range(16):
            devnull = os.open(os.devnull, os.O_WRONLY)
            old_fd = os.dup(2)
            os.dup2(devnull, 2)
            try:
                cap = cv2.VideoCapture(i)
                ok = cap.isOpened()
                cap.release()
            finally:
                os.dup2(old_fd, 2)
                os.close(devnull)
                os.close(old_fd)
            if not ok:
                break
            indexes.append(i)
        for n, idx in enumerate(indexes):
            label = names[n] if n < len(names) else f"Camera {idx}"
            self.camera_combo.addItem(label, idx)
        if not indexes:
            self.camera_combo.addItem(t("no_camera"), -1)
        self.camera_combo.blockSignals(False)

    def _camera_names(self):
        import subprocess
        try:
            r = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-PnpDevice -PresentOnly -Status OK | Where-Object Class -in Camera,Image | Select-Object -ExpandProperty FriendlyName"],
                capture_output=True, text=True, timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
        except Exception:
            return []

    # ─── Screen Capture ───

    def _toggle_screen_settings(self):
        if self._screen_capture and self._screen_capture.isRunning():
            self._stop_screen_capture()
            return
        visible = not self.screen_settings.isVisible()
        if visible:
            from utils.screen_capture import get_monitors
            self.monitor_combo.clear()
            monitors = get_monitors()
            for m in monitors:
                self.monitor_combo.addItem(m["name"], m["index"])
            if not monitors:
                QMessageBox.warning(self, t("title"), t("no_monitor"))
                return
            self._screen_region = None
            self.region_label.setText(t("full_screen"))
        self.screen_settings.setVisible(visible)

    def _start_screen_capture(self):
        if not self._model_path:
            QMessageBox.warning(self, t("title"), t("no_model_loaded"))
            return

        from utils.screen_capture import ScreenCaptureThread

        monitor_idx = self.monitor_combo.currentData()
        fps = self.fps_spin.value()

        self._screen_capture = ScreenCaptureThread(
            monitor_index=monitor_idx, fps=fps, region=self._screen_region
        )
        self._screen_capture.frame_ready.connect(self._on_screen_frame)
        self._screen_capture.log.connect(self._log)
        self._screen_capture.finished.connect(self._on_screen_capture_stopped)
        self._screen_capture.start()

        self.screen_btn.setText(t("stop_capture"))
        self.screen_btn._base_color = "#ef4444"
        self.screen_btn.setStyleSheet(f"""
            HoverButton {{ background: #ef4444; color: #fff; border: none;
            padding: 8px 18px; border-radius: 8px; font-size: 12px; font-weight: bold; }}
        """)
        self.start_screen_btn.setEnabled(False)
        self.status_label.setText(t("screen_capture") + "...")
        self.status_label.setStyleSheet("color: #eab308; font-size: 12px; font-weight: bold;")

        # Start detection server
        self._start_detect_server()

    def _stop_screen_capture(self):
        if self._screen_capture:
            self._screen_capture.stop()
            self._screen_capture.wait(2000)
            self._screen_capture = None

        # Screen capture also owns the detect server — shut it down too
        if self._detect_proc:
            try:
                self._detect_proc.stdin.write(b'{"action":"quit"}\n')
                self._detect_proc.stdin.flush()
                self._detect_proc.wait(timeout=3)
            except Exception:
                try:
                    self._detect_proc.kill()
                except Exception:
                    pass
            self._detect_proc = None

        self.screen_btn.setText(t("screen_capture"))
        self.screen_btn._base_color = "#eab308"
        self.screen_btn.setStyleSheet(f"""
            HoverButton {{ background: #eab308; color: #fff; border: none;
            padding: 8px 18px; border-radius: 8px; font-size: 12px; font-weight: bold; }}
        """)
        self.start_screen_btn.setEnabled(True)
        self.screen_settings.setVisible(False)
        self.status_label.setText(t("ready"))
        self.status_label.setStyleSheet("color: #6b7280; font-size: 12px;")

    def _on_screen_capture_stopped(self):
        self._stop_screen_capture()

    def _select_region(self):
        from utils.region_selector import RegionSelector
        self._region_selector = RegionSelector()
        self._region_selector.region_selected.connect(self._on_region_selected)
        self._region_selector.cancelled.connect(lambda: setattr(self, '_region_selector', None))

    def _on_region_selected(self, x, y, w, h):
        self._screen_region = (x, y, w, h)
        self.region_label.setText(f"Region: {w}x{h} at ({x},{y})")
        self._region_selector = None

    def _on_screen_frame(self, frame):
        """Handle screen capture frame - display and detect."""
        import cv2
        import base64
        import json

        self._last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 else frame
        self._show_frame(self._last_frame)

        # Run detection if server is available
        if self._detect_proc:
            try:
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                frame_b64 = base64.b64encode(buf).decode('utf-8')
                cmd = json.dumps({"action": "detect", "frame": frame_b64,
                    "conf": self.conf_spin.value(), "iou": self.iou_spin.value()}) + "\n"
                self._detect_proc.stdin.write(cmd.encode())
                self._detect_proc.stdin.flush()
                resp_line = self._detect_proc.stdout.readline()
                if resp_line:
                    resp = json.loads(resp_line.decode())
                    if resp.get("status") == "ok" and resp.get("frame"):
                        frame_bytes = base64.b64decode(resp["frame"])
                        nparr = np.frombuffer(frame_bytes, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if img is not None:
                            self._show_frame(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            except:
                pass

    def _on_frame(self, frame):
        self._last_frame = frame
        self._show_frame(frame)

    def _on_dets(self, dets):
        self._all_detections.extend(dets)
        self._last_dets = dets

    def _on_progress(self, cur, total):
        if total > 0:
            self.progress_bar.setValue(int(cur / total * 100))

    def _on_done(self, ok):
        self.progress_bar.setValue(100 if ok else 0)
        # Clean up temp output directory
        import shutil
        import tempfile
        temp_dir = os.path.join(tempfile.gettempdir(), "yolo_detect_output")
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass
        if ok:
            self.status_label.setText(t("detect_done"))
            self.status_label.setStyleSheet("color: #22c55e; font-size: 12px; font-weight: bold;")
        else:
            self.status_label.setText(t("detect_fail"))
            self.status_label.setStyleSheet("color: #ef4444; font-size: 12px; font-weight: bold;")

    def _show_frame(self, frame):
        import cv2
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb = frame
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        data = rgb.copy()
        qimg = QImage(data.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())
        scaled = pixmap.scaled(self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.preview_label.setText("")
        self.preview_label.setPixmap(scaled)

    def _log(self, msg):
        self.status_label.setText(msg)

    def _export_csv(self):
        if not self._all_detections:
            QMessageBox.warning(self, t("title"), t("no_result"))
            return
        filepath, _ = QFileDialog.getSaveFileName(self, t("export_csv"), "detections.csv", "CSV (*.csv)")
        if not filepath:
            return
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["frame", "class", "confidence", "x1", "y1", "x2", "y2"],
                restval="",
                extrasaction="ignore",
            )
            writer.writeheader()
            for det in self._all_detections:
                writer.writerow(det)
        self.status_label.setText(f"CSV: {filepath}")

    def _save_last_annotated(self):
        if not hasattr(self, '_last_frame') or self._last_frame is None:
            QMessageBox.warning(self, t("title"), t("no_result"))
            return
        folder = QFileDialog.getExistingDirectory(self, t("save_annotated"))
        if not folder:
            return
        import cv2
        filename = f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(folder, filename)
        frame = self._last_frame
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        cv2.imwrite(filepath, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.status_label.setText(f"Saved: {filepath}")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if Path(url.toLocalFile()).suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}:
                    event.acceptProposedAction()
                    return
            event.ignore()

    def closeEvent(self, event):
        self._stop_live()
        self._stop_screen_capture()
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
        super().closeEvent(event)

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}:
                self._detect_video_from_path(str(path))
                event.acceptProposedAction()
                return
        event.ignore()
