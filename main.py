import cv2
import platform
import threading
import queue
from typing import Optional

from yolo_server.detector import Detector
from _utils.stream import FrameParser
from config import config

# Global flag to signal all threads to stop
stop_flag = threading.Event()


def display_thread(frame_queue: queue.Queue) -> None:
    if platform.system() == 'Windows':
        cv2.namedWindow("YOLO Detections", cv2.WINDOW_AUTOSIZE)
    elif platform.system() == 'Linux':
        cv2.namedWindow("YOLO Detections", cv2.WINDOW_NORMAL)
    else:
        cv2.namedWindow("YOLO Detections", cv2.WINDOW_NORMAL)

    while not stop_flag.is_set():
        try:
            frame: Optional[cv2.Mat] = frame_queue.get(timeout=1)
            if frame is None:
                break
            cv2.imshow("YOLO Detections", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_flag.set()  # Signal all threads to stop
                break
        except queue.Empty:
            continue
    cv2.destroyAllWindows()


def main() -> None:
    # Initialize the detector
    detector: Detector = Detector(config)

    # Initialize the frame parser
    frame_parser: FrameParser = FrameParser(
        fps=config.fps,
        h=config.height,
        w=config.width,
        source=config.source
    )

    # Create a queue for frames
    frame_queue: queue.Queue = queue.Queue(maxsize=5)  # Limit queue size

    # Start the display thread
    display_thread_handle: threading.Thread = threading.Thread(
        target=display_thread, args=(frame_queue,)
    )
    display_thread_handle.start()

    try:
        while not stop_flag.is_set():
            # Get frame from frame parser
            frame: Optional[cv2.Mat] = frame_parser.get_frame()
            if frame is None:
                print("Failed to capture frame. Exiting...")
                break

            # Perform detection
            bboxes, classes, scores = detector.detect(frame)

            # Draw detections on the frame
            frame_with_detections: cv2.Mat = detector.draw_detections(frame, bboxes, classes, scores)

            # Add frame to queue, skip if queue is full
            try:
                frame_queue.put(frame_with_detections, block=False)
            except queue.Full:
                pass

    except KeyboardInterrupt:
        print("Interrupted by user. Exiting...")
    finally:
        stop_flag.set()  # Signal all threads to stop
        frame_queue.put(None)  # Signal display thread to exit
        display_thread_handle.join(timeout=5)  # Wait for display thread to finish with a timeout
        frame_parser.captured.release()  # Release the video capture object
        cv2.destroyAllWindows()  # Ensure all windows are closed

if __name__ == "__main__":
    main()
    print("Program exited successfully.")