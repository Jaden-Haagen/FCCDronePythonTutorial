import cv2
import mediapipe as mp
import time
from pathlib import Path
from pysimverse import Drone
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core import vision_task_running_mode


MODEL_FILE = Path(__file__).resolve().parent / "models" / "pose_landmarker.task"


def frame_to_mp_image(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)


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
        running_mode=vision_task_running_mode.VisionTaskRunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return vision.PoseLandmarker.create_from_options(options)


# Jump detection method: Velocity-based on hip center (average of left and right hips)
def detect_jump(history):
    if len(history) < 2:
        return False
    prev_ts, prev_lm = history[-2]
    curr_ts, curr_lm = history[-1]
    dt = (curr_ts - prev_ts) / 1000.0
    if dt <= 0:
        return False
    # Check visibility of key landmarks to ensure torso is detected
    hip_visibility = (prev_lm[23].visibility + prev_lm[24].visibility + curr_lm[23].visibility + curr_lm[24].visibility) / 4
    ankle_visibility = (prev_lm[27].visibility + prev_lm[28].visibility + curr_lm[27].visibility + curr_lm[28].visibility) / 4
    if hip_visibility < 0.7 or ankle_visibility < 0.7:
        return False  # Require high visibility to avoid false positives when torso is partially out of frame
    prev_hip_y = (prev_lm[23].y + prev_lm[24].y) / 2
    curr_hip_y = (curr_lm[23].y + curr_lm[24].y) / 2
    dy = curr_hip_y - prev_hip_y
    velocity = dy / dt
    # Consider other body parts: also check ankle velocity, but only count if hips register jump
    prev_ankle_y = (prev_lm[27].y + prev_lm[28].y) / 2
    curr_ankle_y = (curr_lm[27].y + curr_lm[28].y) / 2
    ankle_dy = curr_ankle_y - prev_ankle_y
    ankle_velocity = ankle_dy / dt
    # Stricter thresholds to reduce sensitivity
    return velocity < -0.3 and ankle_velocity < -0.2


def run_jump_detection_camera(camera_id: int = 0):
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}.")

    drone = Drone()
    drone.connect()
    drone.take_off()

    detector = create_pose_landmarker(MODEL_FILE)
    history = []  # list of (timestamp_ms, landmarks)
    jump_counter = 0
    last_jump_time = 0
    jump_phase = None
    jump_start_time = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            timestamp_ms = int(time.time() * 1000)
            image = frame_to_mp_image(frame)
            detection_result = detector.detect_for_video(image, timestamp_ms)

            up_down = 0
            if jump_phase == 'up':
                if timestamp_ms - jump_start_time < 1000:
                    up_down = 50  # move up
                else:
                    jump_phase = 'down'
                    jump_start_time = timestamp_ms
            elif jump_phase == 'down':
                if timestamp_ms - jump_start_time < 1000:
                    up_down = -50  # move down to return to start height
                else:
                    jump_phase = None

            drone.send_rc_control(0, 0, up_down, 0)

            if detection_result.pose_landmarks:
                landmarks = detection_result.pose_landmarks[0]
                history.append((timestamp_ms, landmarks))
                if len(history) > 30:
                    history.pop(0)

                # Detect jumping based on hip center velocity
                jump_detected = detect_jump(history)

                # Update counter with buffer to prevent multiple counts
                if jump_detected and (timestamp_ms - last_jump_time) > 1000:  # 1 second buffer
                    jump_counter += 1
                    last_jump_time = timestamp_ms
                    if jump_phase is None:
                        jump_phase = 'up'
                        jump_start_time = timestamp_ms

                draw_pose_landmarks(frame, landmarks)

                # Display jump detection results
                text = f"Jumps detected: {jump_counter}"
                if jump_detected:
                    text += " - Jumping!"
                else:
                    text += " - Not jumping"

                cv2.putText(
                    frame,
                    text,
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
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

            cv2.imshow("Jump Detection", frame)
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
        run_jump_detection_camera()
    except Exception as exc:
        print(f"Error: {exc}")
