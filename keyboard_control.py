from pysimverse import Drone
import time
import keyboard


MIN_RC = -100
MAX_RC = 100
SPEED = 50
YAW_SPEED = 5

def main():
    drone = Drone()
    drone.connect()
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

    drone.land()
    time.sleep(1)


if __name__ == "__main__":
    main()