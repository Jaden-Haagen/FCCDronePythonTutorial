import time

import cv2
import mediapipe as mp
from pathlib import Path
from pysimverse import Drone
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core import vision_task_running_mode


MODEL_FILE = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"


def frame_to_mp_image(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)


def draw_hand_landmarks(frame, landmarks):
    height, width, _ = frame.shape
    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    ]
    for start, end in connections:
        start_point = (int(landmarks[start].x * width), int(landmarks[start].y * height))
        end_point = (int(landmarks[end].x * width), int(landmarks[end].y * height))
        cv2.line(frame, start_point, end_point, (255, 0, 0), 2)


def draw_position_zones(frame, left_threshold: float, right_threshold: float):
    height, width, _ = frame.shape
    left_x = int(width * left_threshold)
    right_x = int(width * right_threshold)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (left_x, height), (0, 0, 255), -1)
    cv2.rectangle(overlay, (left_x, 0), (right_x, height), (50, 50, 50), -1)
    cv2.rectangle(overlay, (right_x, 0), (width, height), (0, 255, 0), -1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    cv2.line(frame, (left_x, 0), (left_x, height), (255, 255, 255), 2)
    cv2.line(frame, (right_x, 0), (right_x, height), (255, 255, 255), 2)

    cv2.putText(frame, "LEFT", (int(left_x / 2) - 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "CENTER", (int((left_x + right_x) / 2) - 60, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(frame, "RIGHT", (int((right_x + width) / 2) - 40, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


def create_hand_landmarker(model_path: Path, num_hands: int = 1):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}.\n"
            "Download hand_landmarker.task and place it in the models folder."
        )

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision_task_running_mode.VisionTaskRunningMode.IMAGE,
        num_hands=num_hands,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return vision.HandLandmarker.create_from_options(options)


def get_hand_center_x(landmarks):
    # Use the wrist landmark (index 0) as the center
    return landmarks[0].x


def run_hand_position_drone_control(camera_id: int = 0, left_threshold: float = 0.3, right_threshold: float = 0.7):
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}.")

    drone = Drone()
    drone.connect()
    drone.take_off()

    detector = create_hand_landmarker(MODEL_FILE)
    left_right = 0
    forward_backward = 0
    up_down = 0
    yaw = 0
    speed = 80

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            draw_position_zones(frame, left_threshold, right_threshold)
            image = frame_to_mp_image(frame)
            detection_result = detector.detect(image)

            position = None
            left_right = 0
            if detection_result.hand_landmarks:
                hand_landmarks = detection_result.hand_landmarks[0]
                handedness = detection_result.handedness[0][0].display_name

                current_x = get_hand_center_x(hand_landmarks)

                if current_x < left_threshold:
                    position = "left"
                    left_right = -speed
                elif current_x > right_threshold:
                    position = "right"
                    left_right = speed
                # Deadzone in the middle keeps left_right at 0

                drone.send_rc_control(left_right, forward_backward, up_down, yaw)

                draw_hand_landmarks(frame, hand_landmarks)
                cv2.putText(
                    frame,
                    f"{handedness} Hand",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                if position:
                    cv2.putText(
                        frame,
                        f"Position: {position}",
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    print(position)
            else:
                drone.send_rc_control(0, forward_backward, up_down, yaw)
                cv2.putText(
                    frame,
                    "No hand detected",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Hand Position Detection", frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    finally:
        drone.send_rc_control(0, 0, 0, 0)
        drone.land()
        time.sleep(1)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        run_hand_position_drone_control()
    except Exception as exc:
        print(f"Error: {exc}")
