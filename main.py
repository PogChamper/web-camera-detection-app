import cv2
from typing import Optional

from yolo_server.detector import Detector
from _utils.stream import FrameParser
from config import config

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

    # Create a window for displaying the output
    cv2.namedWindow("YOLO Detections", cv2.WINDOW_NORMAL)

    try:
        while True:
            # Get frame from frame parser
            frame: Optional[cv2.Mat] = frame_parser.get_frame()
            if frame is None:
                print("Failed to capture frame. Exiting...")
                break
            frame = cv2.flip(frame, 1)

            # Perform detection
            bboxes, classes, scores = detector.detect(frame)

            # Draw detections on the frame
            frame_with_detections: cv2.Mat = detector.draw_detections(frame, bboxes, classes, scores)

            # Display the frame
            cv2.imshow("YOLO Detections", frame_with_detections)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user. Exiting...")
    finally:
        frame_parser.captured.release()  # Release the video capture object
        cv2.destroyAllWindows()  # Ensure all windows are closed

if __name__ == "__main__":
    main()
    print("Program exited successfully.")
