from pysimverse import Drone
import time


drone = Drone()
drone.connect()
drone.take_off()
drone.set_speed(50)

drone.move_forward(250)
drone.move_right(230)


drone.land()
time.sleep(1)