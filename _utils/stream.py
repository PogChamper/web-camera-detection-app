import os
import time 

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

class Writer:
    def __init__(self, out_directory, fps, h, w, main_video_only=True):
        assert os.path.isdir(out_directory), out_directory
        self.out_directory = out_directory

        self.w = w
        self.h = h
        self.fps = fps

        self.main_video_only = main_video_only

        self.is_init_array = True

        self.name = time.strftime('%Y%m%d_%H%M%S_') + ''.join(hex(np.random.randint(16))[2:] for e in range(4))

    def initialize(self, frames):
        n_frames = len(frames)

        # create writes
        self.writers = []
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if not self.main_video_only:
            if n_frames > 1:
                for i in range(n_frames):
                    out_path = os.path.join(self.out_directory,  f'{self.name}_debug_{i}.avi')
                    self.writers.append(cv2.VideoWriter(out_path, fourcc, self.fps, (self.w, self.h)))
        out_path = os.path.join(self.out_directory, f'{self.name}_debug_general.avi')
        self.writers.append(cv2.VideoWriter(out_path, fourcc, self.fps, (self.w, self.h)))

        # columns rows
        if n_frames == 1:
            self.n_columns = 1
        elif 2 <= n_frames <= 4:
            self.n_columns = 2
        elif 5 <= n_frames <= 9:
            self.n_columns = 3
        elif 10 <= n_frames <= 16:
            self.n_columns = 4
        else:
            assert False, 'слишком большое количество фреймов'
        '''
        if n_frames % self.n_columns == 0:
            self.n_rows = n_frames // self.n_columns
        else:
            self.n_rows = n_frames // self.n_columns + 1
        '''
        self.n_rows = self.n_columns

        self.n_frames = n_frames

    def write(self, frames):
        if self.is_init_array:
            self.is_init_array = False
            self.initialize(frames)

        self._write(frames)

    def release(self):
        for writer in self.writers:
            writer.release()

    def _write(self, frames):

        self.frames_to_show = [f.copy() for f in frames]

        if self.n_frames > 1:
            if not self.main_video_only:
                for i in range(self.n_frames):
                    self.writers[i].write(cv2.resize(frames[i], (self.w, self.h)))

            empty_image = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            rows = []
            for i in range(self.n_rows):
                row_images = []
                for j in range(self.n_columns):
                    if frames:
                        row_images.append(cv2.resize(frames.pop(0), (self.w, self.h)))
                    else:
                        row_images.append(empty_image)
                rows.append(np.hstack(row_images))
            general_frame = np.vstack(rows)
        else:
            general_frame = frames[0]

        self.writers[-1].write(cv2.resize(general_frame, (self.w, self.h)))

    def show_recent_frames(self):
        for i, debug_image in enumerate(self.frames_to_show):
            debug_image = cv2.resize(debug_image, (0, 0),
                                     fx=1000/debug_image.shape[1],
                                     fy=1000/debug_image.shape[1])  # TODO: assumed non-vertical stream
            cv2.imshow(f'debug_image_{i}', debug_image)
        key = cv2.waitKey(1)
        return key
