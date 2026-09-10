"""video_recorder.py — capture a QWidget's rendered frames to an H.264 .mp4 file.

Requires 'imageio' + 'imageio-ffmpeg' (not in the base requirements.txt install —
callers should catch ImportError and prompt the user to install them).
"""

import numpy as np
import imageio
from PySide6.QtGui import QImage


def _widget_to_rgb(widget):
    image = widget.grab().toImage().convertToFormat(QImage.Format.Format_RGB888)
    w, h, stride = image.width(), image.height(), image.bytesPerLine()
    buf = np.frombuffer(image.constBits(), dtype=np.uint8, count=stride * h)
    return buf.reshape(h, stride)[:, :w * 3].reshape(h, w, 3).copy()


class VideoRecorder:
    """Grabs a QWidget on demand and appends frames to an H.264 .mp4 file."""

    def __init__(self, widget, path, fps=20):
        self._widget = widget
        self.path = path
        self.frame_count = 0
        # macro_block_size=2 (H.264's real constraint is even width/height,
        # not the default 16) avoids imageio silently padding/cropping frames.
        self._writer = imageio.get_writer(path, fps=fps, codec='libx264',
                                           quality=8, macro_block_size=2)

    def capture_frame(self):
        self._writer.append_data(_widget_to_rgb(self._widget))
        self.frame_count += 1

    def close(self):
        self._writer.close()
