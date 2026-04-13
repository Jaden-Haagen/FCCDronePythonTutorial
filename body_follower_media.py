import cv2
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core import vision_task_running_mode


MODEL_FILE = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker.task"


def frame_to_mp_image(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)


def classify_pose(landmarks):
    visible_count = sum(1 for landmark in landmarks if landmark.visibility > 0.5)
    if visible_count > 25:
        return "Full Body Detected"
    elif visible_count > 15:
        return "Partial Body Detected"
    else:
        return "Minimal Detection"


def draw_pose_landmarks(frame, landmarks):
    height, width, _ = frame.shape
    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (11, 23), (12, 24),
        (23, 24), (23, 25), (25, 27), (27, 29), (29, 31), (24, 26), (26, 28),
        (28, 30), (30, 32)
    ]
    for start, end in connections:
        start_point = (int(landmarks[start].x * width), int(landmarks[start].y * height))
        end_point = (int(landmarks[end].x * width), int(landmarks[end].y * height))
        cv2.line(frame, start_point, end_point, (255, 0, 0), 2)


def create_pose_landmarker(model_path: Path, num_poses: int = 1):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}.\n"
            "Download pose_landmarker.task and place it in the models folder."
        )

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision_task_running_mode.VisionTaskRunningMode.IMAGE,
        num_poses=num_poses,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return vision.PoseLandmarker.create_from_options(options)


def run_body_follower_camera(camera_id: int = 0):
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}.")

    detector = create_pose_landmarker(MODEL_FILE)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            image = frame_to_mp_image(frame)
            detection_result = detector.detect(image)

            if detection_result.pose_landmarks:
                for pose_landmarks in detection_result.pose_landmarks:
                    pose_label = classify_pose(pose_landmarks)
                    draw_pose_landmarks(frame, pose_landmarks)
                    cv2.putText(
                        frame,
                        pose_label,
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
                    "No pose detected",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Body Follower Detection", frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        run_body_follower_camera()
    except Exception as exc:
        print(f"Error: {exc}")
