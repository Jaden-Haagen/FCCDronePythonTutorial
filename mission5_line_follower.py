import cvzone
from cvzone.ColorModule import ColorFinder
import cv2
import time
import numpy as np
from pysimverse import Drone

# Create an instance of the ColorFinder class with trackBar set to False.
myColorFinder = ColorFinder(trackBar=False)

# Initialize the drone
drone = Drone()
drone.connect()
time.sleep(1)
drone.streamon()
time.sleep(1)
drone.take_off(takeoff_height=30)
time.sleep(1)
drone.set_speed(25)

# Custom color values for detecting orange.
# 'hmin', 'smin', 'vmin' are the minimum values for Hue, Saturation, and Value.
# 'hmax', 'smax', 'vmax' are the maximum values for Hue, Saturation, and Value.
hsvVals = {'hmin': 0, 'smin': 66, 'vmin': 0, 'hmax': 179, 'smax': 255, 'vmax': 255}


def clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def get_largest_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def get_lookahead_point(contour: np.ndarray, height: int, lookahead_ratio: float = 0.5):
    target_y = int(height * lookahead_ratio)
    points = contour.reshape(-1, 2)
    matches = points[np.abs(points[:, 1] - target_y) < 10]
    if len(matches) > 0:
        x = int(np.mean(matches[:, 0]))
        return x, target_y
    closest_idx = np.argmin(np.abs(points[:, 1] - target_y))
    return tuple(points[closest_idx])


def compute_rc_control(frame: np.ndarray, lookahead_point):
    if lookahead_point is None:
        return 0, 0
    center_x = frame.shape[1] // 2
    error_x = lookahead_point[0] - center_x
    yaw = int(np.clip(error_x * 0.17, -100, 100))
    forward = 25
    return yaw, forward


def annotate(frame: np.ndarray, contour, lookahead_point):
    if contour is not None:
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
    if lookahead_point is not None:
        cv2.circle(frame, lookahead_point, 8, (0, 0, 255), -1)
        cv2.line(
            frame,
            (frame.shape[1] // 2, frame.shape[0] - 1),
            lookahead_point,
            (255, 0, 0),
            2,
        )
    cv2.line(frame, (frame.shape[1] // 2, 0), (frame.shape[1] // 2, frame.shape[0]), (255, 255, 255), 1)
    return frame


print("Starting continuous line-following control")
while True:
    frame, is_success = drone.get_frame()
    if not is_success or frame is None:
        continue

    imgOrange, mask = myColorFinder.update(frame, hsvVals)
    mask = clean_mask(mask)

    contour = get_largest_contour(mask)
    lookahead_point = None
    if contour is not None and cv2.contourArea(contour) > 400:
        lookahead_point = get_lookahead_point(contour, frame.shape[0], lookahead_ratio=0.5)

    yaw, forward = compute_rc_control(frame, lookahead_point)
    drone.send_rc_control(0, forward, 0, yaw)

    annotated = annotate(frame.copy(), contour, lookahead_point)
    img_stack = cvzone.stackImages([annotated, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)], 2, 0.5)
    cv2.imshow("Image Stack", img_stack)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print("Landing")
drone.land()
time.sleep(1)
cv2.destroyAllWindows()
