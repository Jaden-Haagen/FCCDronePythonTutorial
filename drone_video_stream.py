from pysimverse import Drone
import time
import cv2

drone = Drone()
drone.connect()
time.sleep(1)
drone.streamon()
time.sleep(1)
drone.take_off()

drone.set_speed(50)
while True:
    frame, is_success = drone.get_frame()

    cv2.imshow("Drone Feed", frame)
    cv2.waitKey(1)
    break

drone.land()
time.sleep(1)