import time
from typing import Tuple, List, Optional, Union

import numpy as np

from config import Config
from _runtime.inference import ONNXInference
from _utils.common import smart_resize_t


def nms(
        dets: np.ndarray, 
        scores: np.ndarray, 
        thresh: float
) -> List[int]:
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # get boxes with more ious first

    keep: List[int] = []
    while order.size > 0:
        i = order[0]  # pick maxmum iou box
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)  # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1)  # maxiumum height
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def non_max_suppression(
    prediction: np.ndarray,
    conf_thres: float = 0.7,
    iou_thres: float = 0.45,
    classes: Optional[List[int]] = None,
    agnostic: bool = True,
    max_det: int = 300,
    nc: int = 80,  # number of classes (optional)
    max_time_img: float = 0.05,
    max_nms: int = 30000,
    max_wh: int = 7680
) -> List[np.ndarray]:
    # Checks
    assert 0 <= conf_thres <= 1, (
        f'Invalid Confidence threshold {conf_thres}, '
        'valid values are between 0.0 and 1.0'
    )
    assert 0 <= iou_thres <= 1, (
        f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    )

    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]  # select only inference output

    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index

    xc = np.amax(prediction[:, 4:mi], axis=1) > conf_thres  # candidates

    # Settings
    time_limit = 0.5 + max_time_img * bs  # seconds to quit after

    # shape(1,84,6300) to shape(1,6300,84)
    prediction = prediction.transpose(0, 2, 1)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

    t = time.time()
    output = [np.zeros((0, 6 + nm))] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask, _ = np.split(x, (4, nc, nm), 1)
        conf, j = np.max(cls, axis=1, keepdims=True), np.argmax(cls, axis=1)
        j = j.astype(np.float32)[..., np.newaxis]
        x = np.concatenate(
            (box, conf, j, mask), 1)[conf.reshape(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            # sort by confidence and remove excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class)

        i = nms(boxes, scores, iou_thres)  # NMS

        i = i[:max_det]  # limit detections
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


class YOLOv11:
    def __init__(self, config: Config):
        self.model_name: str = config.model_name
        self.inference: ONNXInference = ONNXInference(self.model_name)
        self.max_batch_size: int = 1
        self.model_inputs: Tuple[int, int, int] = self.inference.get_inputs()
        self.channels: int = self.model_inputs[0]
        self.height: int = self.model_inputs[1]
        self.width: int = self.model_inputs[2]

        self.conf_thresh: float = config.conf_threshold
        self.nms_thresh: float = config.nms_threshold
        self.classes: List[int] = config.classes

    def __call__(
        self, 
        imgs: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        if not imgs:
            return [], [], []

        batch_size: int = len(imgs)
        batch: np.ndarray = np.zeros(
            shape=(batch_size, self.channels, self.height, self.width),
            dtype=np.float32
        )
        scales: np.ndarray = np.zeros(shape=(batch_size), dtype=np.float32)

        for img_ind, img in enumerate(imgs):
            img, img_scale = smart_resize_t(img, self.height, self.width)
            batch[img_ind] = img
            scales[img_ind] = img_scale

        batch /= 255.0

        outputs: Union[np.ndarray, List[np.ndarray]] = self.inference(batch)
        output_bboxes: List[np.ndarray] = []
        output_classes: List[np.ndarray] = []
        output_scores: List[np.ndarray] = []
        output_nms: List[np.ndarray] = non_max_suppression(
            outputs,
            conf_thres=self.conf_thresh,
            iou_thres=self.nms_thresh,
            classes=self.classes
        )

        for out_ind, out in enumerate(output_nms):
            if len(out) == 0:
                output_bboxes.append(np.array([]))
                output_classes.append(np.array([]))
                output_scores.append(np.array([]))
            else:
                output_bboxes.append(out[:, :4] * scales[out_ind])
                output_classes.append(out[:, 5])
                output_scores.append(out[:, 4])

        return output_bboxes, output_classes, output_scores