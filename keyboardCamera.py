from pysimverse import Drone
import time
import keyboard
import cv2
import datetime


MIN_RC = -100
MAX_RC = 100
SPEED = 50
YAW_SPEED = 5

def main():
    drone = Drone()
    drone.connect()
    drone.streamon()
    time.sleep(1)
    drone.take_off()
    drone.set_speed(SPEED)

    print("Keyboard control active")

    while True:
        if keyboard.is_pressed('x'):
            break

        forward_backward = SPEED if keyboard.is_pressed('w') else -SPEED if keyboard.is_pressed('s') else 0
        left_right = SPEED if keyboard.is_pressed('d') else -SPEED if keyboard.is_pressed('a') else 0
        up_down = SPEED if keyboard.is_pressed('up') else -SPEED if keyboard.is_pressed('down') else 0
        yaw = YAW_SPEED if keyboard.is_pressed('right') else -YAW_SPEED if keyboard.is_pressed('left') else 0

        drone.send_rc_control(left_right, forward_backward, up_down, yaw)
        time.sleep(0.05)

        if keyboard.is_pressed('z'):
            frame, is_success = drone.get_frame()
            if is_success:
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"snapshot_{timestamp}.png"
                cv2.imwrite(filename, frame)
                print(f"Snapshot saved as {filename}")

    drone.land()
    time.sleep(1)


if __name__ == "__main__":
    main()