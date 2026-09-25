"""Screen capture utility using mss for fast frame grab."""

from __future__ import annotations

import time
from PyQt6.QtCore import QThread, pyqtSignal


class ScreenCaptureThread(QThread):
    """Background thread that captures screen frames at a target FPS."""

    frame_ready = pyqtSignal(object)  # numpy BGR array
    log = pyqtSignal(str)

    def __init__(self, monitor_index: int = 1, fps: float = 5.0,
                 region: tuple[int, int, int, int] | None = None,
                 max_height: int = 720):
        super().__init__()
        self.monitor_index = monitor_index
        self.fps = fps
        self.region = region  # (x, y, w, h) or None for full monitor
        self.max_height = max_height
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        import mss
        import numpy as np
        import cv2

        try:
            with mss.mss() as sct:
                monitors = sct.monitors
                if self.monitor_index >= len(monitors):
                    self.monitor_index = 1  # fallback to first real monitor

                self.log.emit(f"Capturing monitor {self.monitor_index}: "
                              f"{monitors[self.monitor_index]['width']}x{monitors[self.monitor_index]['height']}")

                interval = 1.0 / self.fps if self.fps > 0 else 0.1

                while not self._stop:
                    start = time.time()

                    if self.region:
                        x, y, w, h = self.region
                        # mss needs physical pixel coords; region may be in logical coords​‌‌​‌​‌​​‌‌​‌​​‌​‌‌​‌​‌​​‌‌​‌​​‌​‌‌​​‌‌​​‌‌‌​‌​‌
                        # Query the screen's devicePixelRatio to convert​‌‌​‌​‌​​‌‌​‌​​‌​‌‌​‌​‌​​‌‌​‌​​‌​‌​‌‌‌‌‌​‌‌​​​‌​
                        from PyQt6.QtGui import QGuiApplication
                        screens = QGuiApplication.screens()
                        dpr = 1.0
                        for s in screens:
                            geo = s.geometry()
                            if geo.x() <= x < geo.x() + geo.width() and geo.y() <= y < geo.y() + geo.height():
                                dpr = s.devicePixelRatio()
                                break
                        grab = {
                            "left": int(x * dpr),
                            "top": int(y * dpr),
                            "width": int(w * dpr),
                            "height": int(h * dpr),
                        }
                    else:
                        grab = monitors[self.monitor_index]

                    img = np.array(sct.grab(grab))
                    # mss returns BGRA, drop alpha channel​‌‌‌‌​​‌​‌​‌‌‌‌‌​​‌​​​​​‌‌‌​​‌‌‌‌​​​‌‌‌​‌​​​‌​‌‌
                    bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                    # Downscale to save CPU/memory
                    h_orig, w_orig = bgr.shape[:2]
                    if h_orig > self.max_height:
                        scale = self.max_height / h_orig
                        new_w = int(w_orig * scale)
                        bgr = cv2.resize(bgr, (new_w, self.max_height),
                                         interpolation=cv2.INTER_AREA)

                    self.frame_ready.emit(bgr)

                    elapsed = time.time() - start
                    sleep_time = max(0, interval - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

        except Exception as e:
            self.log.emit(f"Screen capture error: {e}")
#唧唧复唧唧著‌‌‌​​‌​​‌​‌‌‌‌‌‌‌​​​‌​‌​‌‌‌​​‌‌​‌​​‌‌‌​‌‌​‌‌​​​​

def get_monitors() -> list[dict]:
    """Return list of available monitors: [{index, name, left, top, width, height}, ...]"""
    import mss
    with mss.mss() as sct:
        result = []
        for i, m in enumerate(sct.monitors):
            if i == 0:
                continue  # skip the "all-in-one" virtual monitor
            result.append({
                "index": i,
                "name": f"Monitor {i} ({m['width']}x{m['height']})",
                "left": m["left"], "top": m["top"],
                "width": m["width"], "height": m["height"],
            })
        return result
