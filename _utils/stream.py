import cv2
import numpy as np
from typing import Optional, Union


class FrameParser:
    def __init__(
        self,
        fps: int,
        h: int,
        w: int,
        source: Union[int, str] = 0,
        seed: Optional[int] = None,
        random_shift: bool = False,
        shift: int = 1
    ):
        self.source = source
        self.w = w
        self.h = h
        self.fps = fps
        self.captured: Optional[cv2.VideoCapture] = None
        self.current_index = 0
        self.capture_current()

        if seed is None:
            seed = np.random.randint(2**30)
        self.rs = np.random.RandomState([322820272, seed])
        self.is_random_shift = random_shift
        self.shift = shift

    def capture_current(self) -> None:
        if self.captured is not None:
            self.captured.release()
            self.captured = None

        if isinstance(self.source, str):
            # For IP camera
            self.captured = cv2.VideoCapture(self.source)
        else:
            # For USB camera
            self.captured = cv2.VideoCapture(self.source)

        self.video_fps = self.captured.get(cv2.CAP_PROP_FPS)

    def get_frame(self) -> Optional[np.ndarray]:
        if self.captured is None:
            return None

        skipped_frames = int(self.video_fps / self.fps) - 1
        if self.is_random_shift:
            shift = self.rs.randint(1 + 2 * self.shift) - self.shift
            skipped_frames += shift

        for _ in range(skipped_frames):
            ret, frame = self.captured.read()
            if not ret:
                self.current_index += 1
                self.capture_current()
                return self.get_frame()

        ret, frame = self.captured.read()
        if not ret:
            self.current_index += 1
            self.capture_current()
            return self.get_frame()

        return cv2.resize(frame, (self.w, self.h))
