import cv2
import numpy as np
import time
from typing import List, Tuple

from yolo_server.yolo import YOLOv11
from config import Config

class Detector:
    def __init__(self, config: Config):
        self.config = config
        self.yolo = YOLOv11(config=self.config)
        self.prev_frame_time = 0
        self.current_frame_time = 0

    def detect(
        self, 
        frame: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        bboxes, classes, scores = self.yolo([frame])
        return bboxes[0], classes[0], scores[0]

    def draw_detections(
        self,
        frame: np.ndarray,
        bboxes: List[np.ndarray],
        classes: List[np.ndarray],
        scores: List[np.ndarray]
    ) -> np.ndarray:
        for bbox, cls, score in zip(bboxes, classes, scores):
            if score < self.config.conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            color = self.config.colors[int(cls) % len(self.config.colors)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{self.config.class_names[int(cls)]}: {score:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Calculate and display FPS
        self.current_frame_time = time.time()
        fps = 1 / (self.current_frame_time - self.prev_frame_time)
        self.prev_frame_time = self.current_frame_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame