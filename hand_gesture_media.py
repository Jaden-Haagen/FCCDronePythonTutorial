import cv2
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core import vision_task_running_mode


MODEL_FILE = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"


def frame_to_mp_image(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)


def get_finger_states(landmarks, handedness_label: str):
    tip_indices = [4, 8, 12, 16, 20]
    pip_indices = [3, 6, 10, 14, 18]

    thumb_tip = landmarks[tip_indices[0]]
    thumb_ip = landmarks[pip_indices[0]]
    if handedness_label == "Right":
        thumb_open = thumb_tip.x > thumb_ip.x
    else:
        thumb_open = thumb_tip.x < thumb_ip.x

    finger_states = [thumb_open]
    for tip_index, pip_index in zip(tip_indices[1:], pip_indices[1:]):
        finger_states.append(landmarks[tip_index].y < landmarks[pip_index].y)

    return finger_states


def classify_gesture(finger_states, landmarks):
    thumb, index, middle, ring, pinky = finger_states
    extended = sum(finger_states)

    if extended == 5:
        return "Open Palm"
    if extended == 0:
        return "Fist"
    if index and middle and not ring and not pinky:
        return "Peace"
    if thumb and not index and not middle and not ring and not pinky:
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        return "Thumbs Up" if thumb_tip.y < thumb_mcp.y else "Thumbs Down"
    if index and not middle and not ring and not pinky:
        return "Point"
    return "Unknown Gesture"


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


def create_hand_landmarker(model_path: Path, num_hands: int = 2):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}.\n"
            "Download hand_landmarker.task and place it in the same folder as this script."
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


def run_hand_gesture_camera(camera_id: int = 0):
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}.")

    detector = create_hand_landmarker(MODEL_FILE)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            image = frame_to_mp_image(frame)
            detection_result = detector.detect(image)

            if detection_result.hand_landmarks:
                for hand_landmarks, handedness in zip(detection_result.hand_landmarks, detection_result.handedness):
                    label = handedness[0].display_name
                    finger_states = get_finger_states(hand_landmarks, label)
                    gesture_label = classify_gesture(finger_states, hand_landmarks)

                    draw_hand_landmarks(frame, hand_landmarks)
                    cv2.putText(
                        frame,
                        f"{label}: {gesture_label}",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
            else:
                cv2.putText(
                    frame,
                    "No hands detected",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Hand Gesture Detection", frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        run_hand_gesture_camera()
    except Exception as exc:
        print(f"Error: {exc}")
