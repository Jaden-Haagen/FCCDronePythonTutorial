from pysimverse import Drone
import time


drone = Drone()
drone.connect()
drone.take_off()

drone.set_speed(50)
drone.move_forward(150)
'''time.sleep(1)
drone.move_right(30)
time.sleep(1)
'''
drone.rotate(10)


drone.land()
time.sleep(1)