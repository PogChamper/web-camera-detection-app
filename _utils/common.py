import cv2
import numpy as np

from typing import List, Tuple



def as_bgr_triple(color):
    '''
    Returns the b,g,r triple with each value in the range 0-255.
    Any transparency will be ignored.
    '''

    return (round(color[2] * 255), round(color[1] * 255), round(color[0] * 255))

class Box:
    def __init__(self, array):
        assert len(array) == 4
        self.left = max(int(array[0]), 0)
        self.top = max(int(array[1]), 0)
        self.right = int(array[2])
        self.bottom = int(array[3])

    @classmethod
    def from_points(cls, points: List[dict]):
        return cls([points[0]['x'], points[0]['y'],
                    points[1]['x'], points[1]['y']])

    def to_points(self) -> list:
        return [{'x': self.left, 'y': self.top},
                {'x': self.right, 'y': self.bottom}]

    def is_valid(self, side_length: int = 1) -> bool:
        assert side_length > 0
        return self.height() > side_length and self.width() > side_length

    def to_list(self) -> list:
        return [self.left, self.top, self.right, self.bottom]

    def left_top(self) -> tuple:
        return (self.left, self.top)

    def right_bottom(self) -> tuple:
        return (self.right, self.bottom)

    def area(self) -> int:
        return (self.right - self.left) * (self.bottom - self.top)

    def center(self) -> np.ndarray:
        return np.array([(self.right + self.left) / 2,
                         (self.bottom + self.top) / 2],
                        dtype=np.float32)

    def height(self) -> int:
        return self.bottom - self.top

    def width(self) -> int:
        return self.right - self.left

    def rescale(self, fx: float, fy: float, max_shape: tuple = None):
        self.left = int(max(self.left * fx, 0))
        self.top = int(max(self.top * fy, 0))
        self.right = int(self.right * fx)
        self.bottom = int(self.bottom * fy)
        if max_shape:
            self.right = min(self.right, max_shape[1] - 1)
            self.bottom = min(self.bottom, max_shape[0] - 1)

    def expand(self, scale: float = 1.0):
        assert scale >= 1.0
        c_x = (self.right + self.left) / 2
        c_y = (self.bottom + self.top) / 2
        r_x = self.width() / 2 * scale
        r_y = self.height() / 2 * scale
        self.left = max(int(c_x - r_x), 0)
        self.top = max(int(c_y - r_y), 0)
        self.right = int(c_x + r_x)
        self.bottom = int(c_y + r_y)

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.top:self.bottom, self.left:self.right].copy()

    def contains(self, x: float, y: float) -> bool:
        return (self.left <= x <= self.right) and \
               (self.top <= y <= self.bottom)

    def draw_on(self,
                frame: np.ndarray,
                border_color: tuple = (0.0, 0.59, 1.0, 1.0),
                fill_color: tuple = (0.0, 0.0, 0.0, 0.0),
                border_width: int = 5) -> np.ndarray:
        border_opacity = border_color.opacity()
        fill_opacity = fill_color.opacity()

        if fill_opacity > 0.0:
            bgr_fill_color = as_bgr_triple(fill_color)
            fill_mask = cv2.rectangle(
                img=frame.copy(),
                pt1=self.left_top(),
                pt2=self.right_bottom(),
                color=bgr_fill_color,
                thickness=cv2.FILLED,
                lineType=cv2.LINE_AA)
            frame = cv2.addWeighted(
                frame,
                1.0 - fill_opacity,
                fill_mask,
                fill_opacity,
                1.0)

        if border_opacity > 0.0:
            bgr_border_color = as_bgr_triple(border_color)
            border_mask = cv2.rectangle(
                img=frame.copy(),
                pt1=self.left_top(),
                pt2=self.right_bottom(),
                color=bgr_border_color,
                thickness=border_width,
                lineType=cv2.LINE_AA)
            frame = cv2.addWeighted(
                frame,
                1.0 - border_opacity,
                border_mask,
                border_opacity,
                1.0)

        return frame

    def wh_aspect_ratio(self) -> float:
        return self.width() / self.height()

    def distance(self, other_box) -> float:
        return np.linalg.norm(self.center() - other_box.center())


def box_iou(box1: Box, box2: Box) -> float:
    '''
    Calculate the Intersection over Union (IoU) of two bounding boxes
    '''

    # determine the coordinates of the intersection rectangle
    x_left = max(box1.left, box2.left)
    y_top = max(box1.top, box2.top)
    x_right = min(box1.right, box2.right)
    y_bottom = min(box1.bottom, box2.bottom)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = box1.area()
    bb2_area = box2.area()

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def mean_box(box1: Box, box2: Box) -> Box:
    return Box([(box1.left + box2.left) // 2,
                (box1.top + box2.top) // 2,
                (box1.right + box2.right) // 2,
                (box1.bottom + box2.bottom) // 2])


def smart_resize_t(img: np.ndarray, height: int, width: int) \
        -> Tuple[np.ndarray, float]:
    '''
    Smart resize which doesnt change aspect ratio

    ---
    Return:

    Tuple[np.ndarray, float] - rescaled image (channel first axis), scale
    '''
    h, w, c = img.shape
    result = np.zeros(shape=(c, height, width), dtype=np.float32)

    if h <= height and w <= width:
        # without resize
        result[:, :h, :w] = np.transpose(img, (2, 0, 1))
        scale = 1.0
    else:
        h_scale = height / h
        w_scale = width / w
        scale = min(h_scale, w_scale)
        resized = cv2.resize(
            img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        new_h, new_w, _ = resized.shape
        result[:, :new_h, :new_w] = np.transpose(resized, (2, 0, 1))
        scale = 1.0 / scale  # inv scale

    return result, scale
